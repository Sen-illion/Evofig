#!/usr/bin/env python3
"""Run one arm of the frozen experiment protocol (server-side batch driver).

Every switch this driver passes comes from `configs/experiment_protocol.json`: the
generation settings, the two Judge seats, the paper list and the arm's own delta. Nothing
about the configuration is written down here, so editing this file cannot change what an
arm means; and each run writes the protocol's sha256 beside its output, so editing the
protocol cannot silently change what an already-finished arm meant either.

An arm whose protocol entry says `blocked` will not run, and one that says
`awaiting_go` only runs when the caller passes `--confirm-go`. That is the protocol
carrying the rule that a new row starts only after a human has looked at its plan.

Three drivers. ``editable`` runs each paper once, in parallel, and promotes nothing.
``session`` walks an ordered queue one station at a time and lets a round's candidate be
judged before the next paper starts, which is the fastest a paper can see what the papers
before it earned. ``waves`` is the middle: a group of papers runs in parallel on one library
snapshot, whatever the group earned is promoted at the group's end, and the next group reads
that. A paper therefore sees everything promoted before its own wave started and nothing its
own wave earned, which is the interleaving the self-evolution rows are about.

Usage (after injecting API credentials):

    <DATA_ROOT>/envs/paperimage/bin/python scripts/run_ablation_arm.py --arm unit_cost
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "configs" / "experiment_protocol.json"

#: The checkout this driver lives in has to be importable ahead of anything else: a driver run
#: against a tree other than its own would read the library with other code than the one that
#: wrote it.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfs.evolution.direct import ExperienceStore  # noqa: E402 - the path has to be set first
from rfs.utils import write_json  # noqa: E402 - the path has to be set first

#: The applicability values a candidate has to carry before the ladder can examine it. The same
#: two the session runner accepts, so a row is queued or dropped here for the same reason.
EXAMINABLE_APPLICABILITY = ("common", "domain")

#: Papers per wave when the arm names no size. One wave shares one library snapshot: everything
#: promoted before it started is visible to all of it, and what it earns is visible from the
#: next wave on.
DEFAULT_WAVE_SIZE = 50

#: The generation settings the product CLI takes as `--name value`, in a stable order.
CLI_SETTINGS = (
    "planner_mode",
    "read_mode",
    "method_mode",
    "structure_mode",
    "text_plan_mode",
    "ppt_plan_mode",
    "color_theme_mode",
    "visual_review_mode",
    "judge_mode",
    "reference_mode",
    "icon_mode",
    "graph_context",
    "middle_layer",
    "figure_target",
    "domain_profile",
    "aspect_ratio",
    "max_rounds",
    "candidates_per_round",
    "icon_candidates",
    "vlm_timeout",
    "vlm_retries",
    "evolution_scope",
    "reference_top_k",
    "reuse_replayed_icons",
)

#: The same settings, under the names a session config uses for its evolve stations.
SESSION_SETTINGS = CLI_SETTINGS + ("evolution_enabled", "experience_enabled")

LOCK = threading.Lock()


def load_protocol(path: Path) -> tuple[dict, str]:
    """Read the protocol and the digest that names exactly the bytes an arm was run on."""

    raw = path.read_bytes()
    body = json.loads(raw.decode("utf-8"))
    if not isinstance(body, dict):
        raise SystemExit("the protocol must be a JSON object: %s" % path)
    return body, hashlib.sha256(raw).hexdigest()


def host_section(protocol: dict, *, key: str = "", hostname: str = "") -> tuple[str, dict]:
    """Pick this machine's block out of the protocol, and the name it is filed under.

    The name matters as much as the block: the Judge holdout lists and the paper lists are
    filed per host, and a run that read the other machine's paths would either refuse to
    start or measure against a tree that is not there.
    """

    hosts = protocol.get("hosts") or {}
    if key:
        if key not in hosts:
            raise SystemExit("the protocol names no host %r; it has %s" % (key, ", ".join(sorted(hosts))))
        return key, dict(hosts[key])
    name = hostname or socket.gethostname()
    for host_key, host in hosts.items():
        if str(host.get("hostname") or "") == name:
            return str(host_key), dict(host)
    raise SystemExit(
        "%s is not one of the machines this protocol names (%s); pass --host with one of those keys"
        % (name, ", ".join(sorted(hosts)))
    )


def arm_by_id(protocol: dict, arm_id: str) -> dict:
    for arm in protocol.get("arms") or []:
        if str(arm.get("id") or "") == arm_id:
            return dict(arm)
    raise SystemExit(
        "the protocol has no arm %r; it has %s"
        % (arm_id, ", ".join(str(item.get("id")) for item in protocol.get("arms") or []))
    )


def settings_for(protocol: dict, arm: dict) -> dict:
    """The arm's switches: the protocol's generation block with this arm's delta on top."""

    values = dict(protocol.get("generation") or {})
    values.update(arm.get("overrides") or {})
    return values


def from_host(value, host: dict, key: str):
    """Resolve an arm path: the literal string `host` means the run host's own value."""

    if isinstance(value, str) and value.strip().lower() == "host":
        return str(host.get(key) or "")
    return str(value or "")


def holdout_lists(protocol: dict, host_key: str) -> tuple[list, list]:
    """The Judge holdout lists that exist on this machine, and the ones that do not.

    A missing list refuses a run at the far end, so calling one out is not a formality: an
    arm that is required to be guarded and cannot be says so instead of running unguarded.
    The lists are filed per host because the two machines mount different trees; they name
    the same exam papers, so the guard means the same thing on both.
    """

    block = (protocol.get("data") or {}).get("judge_holdout") or {}
    by_host = block.get("lists_by_host") or {}
    if host_key not in by_host:
        raise SystemExit("the protocol files no Judge holdout lists under host %r" % host_key)
    named = [str(item) for item in by_host[host_key] or []]
    present = [item for item in named if os.path.isfile(item)]
    missing = [item for item in named if not os.path.isfile(item)]
    return present, missing


def paper_list_for(host: dict, papers: str) -> str:
    """The list an arm reads: named by the arm, filed under the host that runs it.

    An arm that names a list its host does not file is refused, never quietly handed that
    host's default list: running 500 of the wrong papers under the right row name is a
    measurement nobody would notice was wrong.
    """

    name = str(papers or "")
    filed = host.get("paper_lists") or {}
    if name and name in filed:
        return str(filed[name])
    if name and filed:
        raise SystemExit(
            "host %s files no paper list %r; it has %s"
            % (host.get("hostname"), name, ", ".join(sorted(str(key) for key in filed)))
        )
    return str(host.get("paper_list") or "")


def runtime_env(protocol: dict, arm: dict | None = None) -> dict:
    """The environment every worker runs under, so the last knob is not written in code either.

    A key that starts with an underscore is prose in the protocol, not a variable to export.

    An arm may carry an `env` block of its own for a knob that is true of that row and not of the
    protocol's default. Today that is the icon model: one family of rows is drawn under the cheap
    model while the rows another family already measured stay on the ruler they were drawn with.
    The arm's block is layered last, so the protocol still states the default and the row states
    only what it changed.
    """

    block = protocol.get("runtime_env") or {}
    env = {str(name): str(value) for name, value in block.items() if not str(name).startswith("_")}
    for name, value in ((arm or {}).get("env") or {}).items():
        if str(name).startswith("_"):
            continue
        env[str(name)] = str(value)
    return env


def freest_gpu() -> str:
    """Name the GPU with the most free memory, or nothing when nvidia-smi cannot answer."""

    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    best, best_free = "", -1
    for line in (listing.stdout or "").splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        if parts[1].isdigit() and int(parts[1]) > best_free:
            best, best_free = parts[0], int(parts[1])
    return best


#: A paper loads both judge seats in bfloat16, which measures at 37-43 GB for one process, so an
#: 80 GB card carries two papers and no more. Past that the third paper's advisory seat dies with
#: an out-of-memory error, and because that is recorded as an unavailable judge rather than as a
#: refusal the arm keeps running with a weaker mechanism than the one it claims to measure.
MAX_PROCESSES_PER_GPU = 2

#: What a card carries when the row seats no trained Judge. The two seats are the only part of
#: the chain that loads a local model, so a row that takes them off is bound by the chain
#: itself rather than by two 9B weights, and the two-paper rule would leave three quarters of
#: every card idle while the row waits on the model calls it does make.
MAX_PROCESSES_PER_GPU_WITHOUT_SEATS = 4

#: The Judge-block keys an arm may override, and nothing else. A typo has to be refused rather
#: than swallowed: a row that believes it took the trained seats off and did not is a row
#: measured by a mechanism it does not name, which is the defect this whole block guards.
JUDGE_OVERRIDE_KEYS = (
    "advisory_checkpoint",
    "advisory_base_model",
    "acceptance_checkpoint",
    "acceptance_base_model",
)


def judge_settings(protocol: dict, arm: dict) -> dict:
    """The Judge seats this arm runs: the shared block with the arm's own overrides on top.

    An arm that measures nothing -- a row whose delivery is the figure the round drew and
    nothing the seats decided -- names empty checkpoint paths here and runs the rule seats
    instead, which is what takes a local model off each of its cards. A row that wants the
    trained Judge says nothing and inherits the protocol's block whole.
    """

    override = arm.get("judge") or {}
    unknown = sorted(str(name) for name in override if name not in JUDGE_OVERRIDE_KEYS)
    if unknown:
        raise SystemExit(
            "arm %s overrides Judge keys the driver does not know: %s (known: %s)"
            % (arm.get("id"), ", ".join(unknown), ", ".join(JUDGE_OVERRIDE_KEYS))
        )
    return {**(protocol.get("judge") or {}), **override}


def seats_a_trained_judge(protocol: dict, arm: dict) -> bool:
    """Whether this row seats a trained Judge, i.e. whether it loads a local model at all."""

    judge = judge_settings(protocol, arm)
    return bool(
        str(judge.get("advisory_checkpoint") or "").strip()
        or str(judge.get("acceptance_checkpoint") or "").strip()
    )


def gpu_devices() -> list:
    """Every GPU index this machine can see, in nvidia-smi's order; empty when it cannot say."""

    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in (listing.stdout or "").splitlines() if line.strip().isdigit()]


def gpu_rotation(env: dict, workers: int, with_seats: bool = True) -> list:
    """The card each worker slot runs on, refusing a fan-out that no card can carry.

    An explicit ``CUDA_VISIBLE_DEVICES`` is honoured, so a caller that pinned a card keeps getting
    that card; otherwise every GPU the machine reports is used, round robin. Aiming a whole arm at
    one device is what made a three-worker row fill a single card: two papers already sit at about
    73 GB of an 80 GB device, so the third paper's advisory seat ran out of memory and the row went
    on without a judge instead of refusing.

    ``with_seats`` is whether the row seats a trained Judge. It decides the ceiling, because the
    seats are what the ceiling is made of: a row without them holds nothing on the card worth
    counting, and refusing its fan-out by the seats' rule would only slow it down.
    """

    named = [item.strip() for item in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",") if item.strip()]
    devices = named or gpu_devices()
    if not devices:
        return []
    per_card = MAX_PROCESSES_PER_GPU if with_seats else MAX_PROCESSES_PER_GPU_WITHOUT_SEATS
    ceiling = per_card * len(devices)
    if workers > ceiling:
        if with_seats:
            raise SystemExit(
                "asked for %d workers on %d card(s): one paper loads both judge seats and measures at "
                "37-43 GB, so a card carries at most %d of them. Lower --concurrency to %d, or add cards."
                % (workers, len(devices), per_card, ceiling)
            )
        raise SystemExit(
            "asked for %d workers on %d card(s): this row seats no trained Judge, so a card carries "
            "at most %d papers of the chain itself. Lower --concurrency to %d."
            % (workers, len(devices), per_card, ceiling)
        )
    return devices


def load_env(path: str) -> dict:
    env = os.environ.copy()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            env[name.strip()] = value.strip()
    return env


def require_judge_assets(protocol: dict) -> None:
    """Stop before the first paid call when a Judge seat cannot be seated.

    A missing checkpoint does not fail on its own: the seat answers unavailable and the round
    keeps the heuristic gate findings, so an arm would finish quietly judged by rules instead
    of by the trained Judge. That is a wrong measurement, not a degraded one.
    """

    models = protocol.get("models") or {}
    checkpoint = str(models.get("judge_checkpoint") or "")
    base = str(models.get("judge_base_model") or "")
    missing = [path for path in (checkpoint, base) if not os.path.exists(path)]
    if missing:
        raise SystemExit("the trained Judge cannot be seated; missing: " + ", ".join(missing))
    if not os.path.isfile(os.path.join(checkpoint, "adapter", "adapter_config.json")):
        raise SystemExit("the Judge checkpoint holds no adapter: " + checkpoint)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def read_list(path: str, limit: int, stride: int = 1) -> list:
    """Read up to ``limit`` papers, optionally taking every ``stride``-th one.

    A stride is what keeps a 20-paper price sample from being 20 papers of the first domain:
    stepping through the whole list spans every domain the benchmark holds.
    """

    rows: list = []
    seen = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            index, seen = seen, seen + 1
            if stride > 1 and index % stride:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def replay_prefix(arm: dict, run_root: str, row: dict) -> tuple[str, str]:
    """The prior run one paper of a forked arm inherits its prefix from, and the fork stage.

    The two axes of the inference-budget experiment live at different points in the pipeline,
    so neither can be measured by simply re-running a finished arm. ``fork_before`` names the
    first stage this arm plans for itself; everything before it -- the extraction of the paper,
    its method contract, its scientific graph and its relation explanations -- is inherited
    from the source arm, which makes the arms of one family a controlled pair that differ only
    from the fork stage on. An arm that names no source gets two empty strings and runs whole.
    """

    source_arm = str(arm.get("replay_from_arm") or "")
    if not source_arm:
        return "", ""
    stage = str(arm.get("fork_before") or "")
    if not stage:
        raise SystemExit(
            "arm %s replays from %s but names no fork stage, so its pair would not be controlled"
            % (arm.get("id"), source_arm)
        )
    source = os.path.join(
        run_root, "out", source_arm, safe(str(row.get("domain") or "")), safe(str(row.get("paper_id") or ""))
    )
    return source, stage


def library_for(arm: dict, host: dict) -> str:
    """The reference library an arm reads, resolved against the host that runs it.

    An arm that names one and lands on a host that files none is refused rather than quietly
    run without it: a retrieval-depth rung that read no library would report the default's
    number under the arm's name, which is a measurement nobody would notice was wrong.
    """

    named = arm.get("reference_library")
    if named is None or (isinstance(named, str) and not named.strip()):
        return ""
    library = from_host(named, host, "reference_library")
    if not library:
        raise SystemExit(
            "arm %s reads a reference library but host %s names none"
            % (arm.get("id"), host.get("hostname"))
        )
    return library


def upstream_contract_for(host: dict, row: dict) -> str:
    """The middle layer's directory for one paper, when the host files its graph tree.

    The middle layer states a paper's method in more depth than this run plans for itself,
    and its tree files one directory per paper under a host-level root. A run pointed at
    one of those merges that paper's inventory into the contract it plans itself; a host
    that files no tree, and a paper the tree does not cover, leave the run planning on its
    own exactly as before, so a partly covered list still comes back as one row.
    """

    root = str(host.get("upstream_graphs") or "")
    if not root or not os.path.isdir(root):
        return ""
    names = [
        str(row.get("paper_id") or ""),
        str(row.get("name") or ""),
        os.path.splitext(os.path.basename(str(row.get("pdf_path") or "")))[0],
    ]
    for name in names:
        if not name:
            continue
        for candidate in dict.fromkeys((name, safe(name))):
            directory = os.path.join(root, candidate)
            if os.path.isdir(directory):
                return directory
    return ""


def editable_command(
    protocol: dict, arm: dict, host: dict, paper: str, out: str, present: list,
    replay_from: str = "", fork_before: str = "", upstream_contract: str = "",
) -> list:
    """The product command for one paper, every switch read out of the protocol."""

    values = settings_for(protocol, arm)
    judge = judge_settings(protocol, arm)
    cmd = ["{python}", "-m", "rfs.cli", "paper-evolve-editable", "--paper", paper, "--out", out]
    for name in CLI_SETTINGS:
        if name not in values:
            raise SystemExit("the protocol does not set %s, and the arm does not override it" % name)
        cmd += ["--" + name.replace("_", "-"), str(values[name])]
    cmd += ["--evolution", "on" if values.get("evolution_enabled") else "off"]
    library = from_host(arm.get("experience_library"), host, "experience_library")
    if values.get("experience_enabled"):
        if not library:
            raise SystemExit("arm %s keeps the experience library but this host names none" % arm.get("id"))
        cmd += ["--experience-library", library]
    else:
        cmd += ["--no-experience"]
    library = library_for(arm, host)
    if library:
        cmd += ["--reference-library", library]
    if upstream_contract:
        cmd += ["--upstream-from", upstream_contract]
    if replay_from:
        cmd += ["--replay-upstream-from", replay_from, "--fork-before", fork_before]
    for path in present:
        cmd += ["--judge-holdout", path]
    cmd += [
        "--acceptance-checkpoint", str(judge.get("acceptance_checkpoint") or ""),
        "--acceptance-base-model", str(judge.get("acceptance_base_model") or ""),
        "--advisory-checkpoint", str(judge.get("advisory_checkpoint") or ""),
        "--advisory-base-model", str(judge.get("advisory_base_model") or ""),
        "--json",
    ]
    return cmd


def paper_out(root_out: str, row: dict) -> str:
    """Where one paper's delivery lives, in every driver that runs one paper per call."""

    return os.path.join(root_out, safe(str(row.get("domain") or "")), safe(str(row.get("paper_id") or "")))


def wave_slices(rows: list, size: int) -> list:
    """Cut the paper list into waves of at most ``size``, keeping the order the list has."""

    step = int(size or 0) or DEFAULT_WAVE_SIZE
    if step < 1:
        step = DEFAULT_WAVE_SIZE
    return [rows[index:index + step] for index in range(0, len(rows), step)]


def wave_candidates(rows: list, root_out: str) -> dict:
    """The rows one wave's papers wrote: experience id -> the applicability it carries.

    This reads the file a round records its own candidate in, which is the file the session
    runner reads too, so a candidate is found here for the same reason it is found there. A
    paper that failed wrote nothing, and its absence is not an error: it simply has no row.
    """

    found: dict = {}
    for row in rows:
        for path in sorted(Path(paper_out(root_out, row)).glob("round_*/experience_candidate.json")):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(body, dict):
                continue
            experience_id = str(body.get("experience_id") or "")
            if experience_id:
                found.setdefault(experience_id, str(body.get("applicability") or "").strip().casefold())
    return found


def library_snapshot(library: str) -> dict:
    """The library as a wave boundary finds it: a digest over its bytes and its rows by status."""

    if not library or not Path(library).is_dir():
        return {"path": str(library or ""), "digest": "", "rows": 0, "by_status": {}}
    digest = hashlib.sha256()
    for name in ("manifest.json", "judge_experiences.jsonl", "experience_events.jsonl"):
        path = Path(library) / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    try:
        records = ExperienceStore(library).records_with_status()
    except Exception:  # noqa: BLE001 - an unreadable library must not stop a wave already running
        return {"path": str(library), "digest": digest.hexdigest(), "rows": None, "by_status": {}}
    by_status: dict = {}
    for record in records:
        key = str(record.get("status") or "unknown")
        by_status[key] = by_status.get(key, 0) + 1
    return {"path": str(library), "digest": digest.hexdigest(), "rows": len(records), "by_status": by_status}


def validate_command(protocol: dict, arm: dict, host: dict, experience_id: str, out: str) -> list:
    """The paired OFF/ON call that judges one candidate and promotes or refutes it.

    The Judge seats travel inside the validation suite rather than on this command line: the
    suite is where the checkpoints were written down, and a row judged by a different model
    than the round that earned it would be measuring something else.
    """

    library = from_host(arm.get("experience_library"), host, "experience_library")
    suite = from_host(arm.get("validation_suite"), host, "validation_suite")
    return [
        "{python}", "-m", "rfs.cli", "paper-evolve-validate",
        "--suite", suite,
        "--experience-library", library,
        "--experience-id", experience_id,
        "--out", out,
        "--json",
    ]


def session_config(protocol: dict, arm: dict, rows: list, present: list, session_id: str) -> dict:
    """The session config for an arm: one evolve station per paper, the delta on every one.

    The validate stations are not written here on purpose. The session queues a validation
    for each candidate an evolve station earns, on the budget that is left, which is the
    interleaving this row is about -- writing them by hand would freeze a queue that has to
    follow what the papers actually produced.
    """

    values = settings_for(protocol, arm)
    judge = judge_settings(protocol, arm)
    options: dict = {name: values[name] for name in SESSION_SETTINGS if name in values}
    options["acceptance_checkpoint"] = str(judge.get("acceptance_checkpoint") or "")
    options["acceptance_base_model"] = str(judge.get("acceptance_base_model") or "")
    options["online_judge_checkpoint"] = str(judge.get("advisory_checkpoint") or "")
    options["online_judge_base_model"] = str(judge.get("advisory_base_model") or "")
    if present:
        options["judge_holdout_paths"] = list(present)
    stations = []
    for row in rows:
        station = {"kind": "evolve", "paper": row["pdf_path"], "paper_id": row["paper_id"]}
        station.update(options)
        stations.append(station)
    return {"session_id": session_id, "stations": stations}


def session_command(protocol: dict, arm: dict, host: dict, config_path: str, out: str, present: list) -> list:
    judge = judge_settings(protocol, arm)
    cmd = ["{python}", "-m", "rfs.cli", "paper-evolve-session", "--config", config_path, "--out", out]
    library = from_host(arm.get("experience_library"), host, "experience_library")
    if library:
        cmd += ["--experience-library", library]
    suite = from_host(arm.get("validation_suite"), host, "validation_suite")
    if suite:
        cmd += ["--validation-suite", suite]
    cmd += [
        "--acceptance-checkpoint", str(judge.get("acceptance_checkpoint") or ""),
        "--acceptance-base-model", str(judge.get("acceptance_base_model") or ""),
    ]
    for path in present:
        cmd += ["--judge-holdout", path]
    if not (arm.get("overrides") or {}).get("validate", True):
        cmd += ["--no-validate"]
    cmd += ["--json"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one arm of the frozen experiment protocol.")
    ap.add_argument("--arm", required=True, help="Arm id from the protocol.")
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL), help="Machine-readable protocol JSON.")
    ap.add_argument("--host", default="", help="Host key to use instead of this machine's hostname.")
    ap.add_argument("--code", default="", help="Checkout to run; defaults to the protocol's host block.")
    ap.add_argument("--python", default="", help="Interpreter; defaults to the protocol's host block.")
    ap.add_argument("--env-file", default="", help="Credential file; defaults to the protocol's host block.")
    ap.add_argument("--list", default="", help="Paper list JSONL; defaults to the protocol's host block.")
    ap.add_argument("--out", default="", help="Run root; defaults to the protocol's host block.")
    ap.add_argument("--limit", type=int, default=0, help="Run at most N papers (0 uses the arm's own).")
    ap.add_argument("--stride", type=int, default=0, help="Take every Nth paper of the list (0 uses the arm's own).")
    ap.add_argument("--concurrency", type=int, default=0, help="Workers (0 uses the arm's own).")
    ap.add_argument(
        "--driver", choices=["", "editable", "session", "waves"],
        help="Override the arm's driver. Used to sample one arm per paper for the cost calibration.",
    )
    ap.add_argument("--confirm-go", action="store_true", help="Allow an arm the protocol marks awaiting_go.")
    ap.add_argument("--dry-run", action="store_true", help="Print the command and write nothing.")
    args = ap.parse_args()

    protocol, digest = load_protocol(Path(args.protocol))
    arm = arm_by_id(protocol, args.arm)
    host_key, host = host_section(protocol, key=args.host)
    status = str(arm.get("status") or "")
    driver = args.driver or str(arm.get("driver") or "")
    if status == "blocked":
        raise SystemExit("arm %s is blocked: %s" % (args.arm, arm.get("status_note") or ""))
    if status == "awaiting_go" and not args.confirm_go:
        raise SystemExit(
            "arm %s is ready but waits for a human go: %s. Pass --confirm-go once its plan, hours and cost are approved."
            % (args.arm, arm.get("status_note") or "")
        )
    if driver == "none":
        raise SystemExit("arm %s does not run on its own: %s" % (args.arm, arm.get("status_note") or ""))

    code = args.code or str(host.get("code") or "")
    python = args.python or str(host.get("python") or "")
    env_file = args.env_file or str(host.get("env_file") or "")
    run_root = args.out or str(host.get("run_root") or "")
    paper_list = args.list or paper_list_for(host, arm.get("papers"))
    limit = args.limit or int(arm.get("limit") or 0)
    stride = args.stride or int(arm.get("stride") or 0) or 1
    concurrency = args.concurrency or int(arm.get("concurrency") or 0) or 3
    if driver in ("editable", "waves"):
        # Refuse a fan-out no card can carry before the arm spends anything on it, so that a
        # --dry-run shows the refusal too instead of a plan nobody can run.
        seeded = load_env(env_file) if env_file and os.path.exists(env_file) else os.environ
        gpu_rotation(seeded, concurrency, seats_a_trained_judge(protocol, arm))
    present, missing = holdout_lists(protocol, host_key)

    print("arm=%s driver=%s protocol=%s@%s" % (args.arm, driver, protocol.get("schema"), digest))
    print("code=%s python=%s run_root=%s list=%s limit=%s stride=%s" % (code, python, run_root, paper_list, limit or "all", stride))
    if missing:
        note = "holdout lists not on this machine: " + ", ".join(missing)
        if arm.get("require_holdout"):
            raise SystemExit(note + " -- this arm must be guarded, so it refuses to start")
        print("note: " + note)

    if driver == "waves" and bool((arm.get("overrides") or {}).get("validate", True)) and not from_host(arm.get("validation_suite"), host, "validation_suite"):
        # A wave row is only self-evolving when something is promoted between its waves. Without a
        # suite to judge them on, every wave would read the same library and the row would say
        # nothing about the chain, so it is refused here, before the paper list is even read.
        raise SystemExit(
            "arm %s promotes between its waves and this host names no validation suite: without one "
            "every wave would read the same library and the row would not be self-evolution" % args.arm
        )

    if not args.dry_run and seats_a_trained_judge(protocol, arm):
        require_judge_assets(protocol)
    rows = read_list(paper_list, limit or 10 ** 9, stride)
    root_out = os.path.join(run_root, "out", args.arm)
    os.makedirs(root_out, exist_ok=True)
    logs = os.path.join(run_root, "logs")
    os.makedirs(logs, exist_ok=True)
    log_path = os.path.join(logs, args.arm + ".log")
    stamp = {
        "schema": protocol.get("schema"),
        "version": protocol.get("version"),
        "sha256": digest,
        "arm": args.arm,
        "driver": driver,
        "code": code,
        # What this row inherited and what it read: a reader who finds two arms of one family
        # disagreeing can tell from here whether they were handed the same prefix and the same
        # library, which the switches alone do not say.
        "replay_from_arm": str(arm.get("replay_from_arm") or ""),
        "fork_before": str(arm.get("fork_before") or ""),
        "reference_library": library_for(arm, host),
        # Which Judge measured this row. A row that took the trained seats off is not comparable
        # with one that kept them, and the difference is invisible in the switches above.
        "judge_seats": "trained" if seats_a_trained_judge(protocol, arm) else "heuristic",
        "judge_overrides": {
            str(name): str(value)
            for name, value in (arm.get("judge") or {}).items()
        },
        # The row's own environment, when it differs from the shared block. A row drawn under a
        # different icon model than the protocol names has to say so where the row is filed, or
        # two rulers look like one.
        "env_overrides": {
            str(name): str(value)
            for name, value in (arm.get("env") or {}).items()
            if not str(name).startswith("_")
        },
    }
    if not args.dry_run:
        Path(os.path.join(root_out, "protocol.json")).write_text(
            json.dumps(stamp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def log(message: str) -> None:
        with LOCK:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("%s %s\n" % (time.strftime("%F %T"), message))

    log("start arm=%s driver=%s papers=%d concurrency=%d protocol=%s" % (args.arm, driver, len(rows), concurrency, digest))
    for name, value in sorted((arm.get("env") or {}).items()):
        if not str(name).startswith("_"):
            log("env override %s=%s" % (name, value))
    if not seats_a_trained_judge(protocol, arm):
        log("judge seats off: this row seats the rule Judge, so no local weights load and a card carries up to %d papers" % MAX_PROCESSES_PER_GPU_WITHOUT_SEATS)

    if driver == "session":
        config_path = os.path.join(run_root, "out", args.arm, "session_config.json")
        body = session_config(protocol, arm, rows, present, "%s-%s" % (args.arm, digest[:12]))
        if args.dry_run:
            print("session stations=%d -> %s" % (len(body["stations"]), config_path))
            for line in session_command(protocol, arm, host, config_path, os.path.join(root_out, "session"), present):
                print("  " + line)
            return 0
        Path(config_path).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        env = load_env(env_file)
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = freest_gpu()
        cmd = [item.replace("{python}", python) for item in session_command(protocol, arm, host, config_path, os.path.join(root_out, "session"), present)]
        proc = subprocess.run(cmd, cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        Path(os.path.join(root_out, "session_run.json")).write_text(proc.stdout or "", encoding="utf-8")
        Path(os.path.join(root_out, "session_run.err")).write_text(proc.stderr or "", encoding="utf-8")
        log("session rc=%d" % proc.returncode)
        print("session rc=%d; log %s" % (proc.returncode, log_path))
        return proc.returncode

    def one(index: int, row: dict, total: int, env: dict, device: str = "") -> str:
        out = paper_out(root_out, row)
        if os.path.exists(os.path.join(out, "editable_composition.pptx")):
            return "skip %s" % out
        source, stage = replay_prefix(arm, run_root, row)
        if source and not os.path.isdir(source):
            # The prefix is per paper, so one paper the source arm never delivered is a gap in
            # this arm's prefix rather than a reason to refuse the arm. It is run whole, and the
            # log says so: a pair where one side was forked and the other was not has to be
            # readable afterwards.
            log("[%d/%d] no prefix for %s; running it whole" % (index, total, out))
            source, stage = "", ""
        cmd = [
            item.replace("{python}", python)
            for item in editable_command(
                protocol, arm, host, row["pdf_path"], out, present, replay_from=source, fork_before=stage,
                upstream_contract=upstream_contract_for(host, row),
            )
        ]
        if args.dry_run:
            return "dry-run: " + " ".join(cmd)
        os.makedirs(out, exist_ok=True)
        run_env = dict(env)
        if device:
            run_env["CUDA_VISIBLE_DEVICES"] = device
        started = time.time()
        with open(os.path.join(out, "run.err"), "a", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=code, env=run_env, stdout=subprocess.PIPE, stderr=err, text=True)
            if source and proc.returncode != 0:
                # A forked run can fail because the prefix it was handed cannot carry it -- a
                # stage the pair must inherit whole is missing from the source. That is a
                # property of the source, not of this paper, so the paper is attempted once
                # more from the beginning instead of being lost, and the record keeps both.
                log(
                    "[%d/%d] forked run rc=%d; retrying %s whole" % (index, total, proc.returncode, out)
                )
                cmd = [
                    item.replace("{python}", python)
                    for item in editable_command(
                        protocol, arm, host, row["pdf_path"], out, present, replay_from="", fork_before="",
                        upstream_contract=upstream_contract_for(host, row),
                    )
                ]
                proc = subprocess.run(cmd, cwd=code, env=run_env, stdout=subprocess.PIPE, stderr=err, text=True)
                log("[%d/%d] whole run rc=%d" % (index, total, proc.returncode))
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as fh:
            fh.write(proc.stdout or "")
        took = time.time() - started
        log("[%d/%d] rc=%d %.0fs %s" % (index, total, proc.returncode, took, out))
        return "rc=%d %s" % (proc.returncode, out)

    def run_papers(subset: list, env: dict, devices: list) -> None:
        """Run one group of papers at this arm's concurrency; a failed paper never ends the arm.

        Each worker is handed a card of its own in turn, so a row's fan-out is spread over the
        machine instead of landing on whichever device looked emptiest when the arm started.
        """

        def card(number: int) -> str:
            return devices[(number - 1) % len(devices)] if devices else ""

        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(one, i + 1, r, len(subset), env, card(i + 1)) for i, r in enumerate(subset)
            ]
            for future in cf.as_completed(futures):
                try:
                    print(future.result(), flush=True)
                except Exception as exc:  # noqa: BLE001 - a failed paper must not kill the arm
                    print("ERROR %r" % (exc,), flush=True)

    if driver == "waves":
        size = int(arm.get("wave_size") or 0) or DEFAULT_WAVE_SIZE
        library = from_host(arm.get("experience_library"), host, "experience_library")
        suite = from_host(arm.get("validation_suite"), host, "validation_suite")
        promote = bool((arm.get("overrides") or {}).get("validate", True))
        waves = wave_slices(rows, size)
        waves_root = os.path.join(root_out, "waves")
        before = library_snapshot(library)
        print(
            "waves=%d wave_size=%d concurrency=%d promote=%s library=%s rows=%s"
            % (len(waves), size, concurrency, promote, library, before["rows"])
        )
        if args.dry_run:
            for index, wave in enumerate(waves, 1):
                print("  wave %02d: %d papers, %s .. %s" % (index, len(wave), wave[0]["paper_id"], wave[-1]["paper_id"]))
            if promote:
                plan = validate_command(protocol, arm, host, "<experience-id>", "<out>")
                print("  then promote: " + " ".join(plan).replace("{python}", python))
            else:
                print("  no validation: this arm keeps the library as it finds it")
            return 0
        env = load_env(env_file)
        env.update(runtime_env(protocol, arm))
        devices = gpu_rotation(env, concurrency, seats_a_trained_judge(protocol, arm))
        os.makedirs(waves_root, exist_ok=True)
        log("start waves=%d wave_size=%d library=%s papers=%d concurrency=%d protocol=%s" % (len(waves), size, library, len(rows), concurrency, digest))
        for index, wave in enumerate(waves, 1):
            log("wave %02d start papers=%d library=%s" % (index, len(wave), str(before.get("digest") or "")[:12]))
            run_papers(wave, env, devices)
            found = wave_candidates(wave, root_out)
            examined = sorted(key for key, value in found.items() if value in EXAMINABLE_APPLICABILITY)
            declined = sorted(key for key, value in found.items() if value not in EXAMINABLE_APPLICABILITY)
            validations: list = []
            for experience_id in examined if promote else []:
                out = os.path.join(waves_root, "validate", "%02d_%s" % (index, safe(experience_id)))
                os.makedirs(out, exist_ok=True)
                cmd = [item.replace("{python}", python) for item in validate_command(protocol, arm, host, experience_id, out)]
                started = time.time()
                proc = subprocess.run(cmd, cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                Path(os.path.join(out, "run.json")).write_text(proc.stdout or "", encoding="utf-8")
                Path(os.path.join(out, "run.err")).write_text(proc.stderr or "", encoding="utf-8")
                took = time.time() - started
                log("wave %02d validate rc=%d %.0fs %s" % (index, proc.returncode, took, experience_id))
                validations.append({"experience_id": experience_id, "rc": proc.returncode, "seconds": round(took), "out_dir": out})
            after = library_snapshot(library)
            write_json(
                Path(waves_root) / ("wave_%02d.json" % index),
                {
                    "arm": args.arm,
                    "driver": "waves",
                    "wave": index,
                    "of": len(waves),
                    "wave_size": size,
                    "concurrency": concurrency,
                    "papers": [str(row.get("paper_id") or "") for row in wave],
                    "library_before": before,
                    "library_after": after,
                    "candidates": sorted(found),
                    "examined": examined,
                    "declined": declined,
                    "declined_because": "no applicability was written, so the selector is required to refuse the row",
                    "promoted_between_waves": promote,
                    "validation_suite": suite,
                    "validations": validations,
                },
            )
            log("wave %02d done candidates=%d examined=%d library=%s" % (index, len(found), len(examined), str(after.get("digest") or "")[:12]))
            print(
                "wave %02d/%02d: %d candidates, %d examined; library now %s"
                % (index, len(waves), len(found), len(examined), after.get("by_status")),
                flush=True,
            )
            before = after
        log("done")
        print("arm %s finished; log %s" % (args.arm, log_path))
        return 0

    env = load_env(env_file)
    env.update(runtime_env(protocol, arm))
    devices = gpu_rotation(env, concurrency, seats_a_trained_judge(protocol, arm))
    run_papers(rows, env, devices)
    log("done")
    print("arm %s finished; log %s" % (args.arm, log_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
