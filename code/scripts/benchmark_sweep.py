"""Run paper-evolve-editable over a frozen EditableFigureBench instance set.

Thin, resumable subprocess driver: for every instance directory it runs the real
generation chain once, records wall time, exit code and whether an editable deck
was delivered, and appends one JSON line per instance. It exists so the benchmark
sweep is one reproducible command instead of an ad-hoc shell loop; it is not part
of the product chain and writes nothing into the product contract.

    python scripts/benchmark_sweep.py --instances <instances> --runs <runs> \
        --checkout <checkout> --env-file <dotenv> --workers 4 --resume \
        --experience-mode isolated

Every model-backed stage is passed explicitly and recorded. `paper-evolve-editable`
defaults planner, method extraction and the Judge to their heuristic settings, so a
sweep that relies on the defaults measures the offline-smoke configuration and not
the system. The 2026-09-28 run did exactly that and produced generic template decks
that scored ~0 on content fidelity, which is a property of the configuration rather
than of the system. This driver therefore always passes the product settings, refuses
to start when a model-backed stage has no credentials, and writes the resolved
configuration next to the runs so the numbers stay interpretable.

Why `--experience-library` is always passed: `paper-evolve-editable` defaults the
experience library to `<out_dir>.parent/paper_evolve_experience_library`, so every
instance sharing an output parent would read and write the same library and the
instances would stop being independent. `isolated` (the default here) gives each
instance its own library, `shared` reproduces the accumulating product behaviour,
and `off` disables experience entirely.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

EXPERIENCE_MODES = ("isolated", "shared", "off")

#: Stage settings the *product* runs.
PRODUCT_CONFIGURATION = {
    "planner_mode": "vlm",
    "method_mode": "vlm",
    "structure_mode": "vlm",
    "text_plan_mode": "vlm",
    "ppt_plan_mode": "vlm",
    "color_theme_mode": "vlm",
    "icon_mode": "image2",
    "visual_review_mode": "vlm",
    "judge_mode": "vlm",
}

#: The settings README documents for an offline smoke test: no model, no API.
OFFLINE_SMOKE_CONFIGURATION = {
    "planner_mode": "heuristic",
    "method_mode": "heuristic",
    "structure_mode": "heuristic",
    "text_plan_mode": "heuristic",
    "ppt_plan_mode": "heuristic",
    "color_theme_mode": "fallback",
    "icon_mode": "placeholder",
    "visual_review_mode": "heuristic",
    "judge_mode": "heuristic",
}

#: Accepted values per stage, mirroring `rfs paper-evolve-editable`.
CONFIGURATION_CHOICES = {
    "planner_mode": ("vlm", "heuristic"),
    "method_mode": ("heuristic", "vlm"),
    "structure_mode": ("vlm", "heuristic"),
    "text_plan_mode": ("vlm", "heuristic"),
    "ppt_plan_mode": ("vlm", "heuristic"),
    "color_theme_mode": ("vlm", "fallback"),
    "icon_mode": ("placeholder", "image2"),
    "visual_review_mode": ("auto", "off", "heuristic", "vlm"),
    "judge_mode": ("heuristic", "vlm", "trained"),
}

#: Settings that never call a model, so they need no credentials.
LOCAL_SETTINGS = {
    "planner_mode": ("heuristic",),
    "method_mode": ("heuristic",),
    "structure_mode": ("heuristic",),
    "text_plan_mode": ("heuristic",),
    "ppt_plan_mode": ("heuristic",),
    "color_theme_mode": ("fallback",),
    "icon_mode": ("placeholder",),
    "visual_review_mode": ("off", "heuristic"),
    "judge_mode": ("heuristic",),
}

#: Credentials a model-backed sweep needs before it may start.
REQUIRED_CREDENTIALS = ("API_BASE", "API_KEY")


def parse_args(argv=None):
    """Parse the sweep command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", required=True, help="frozen instance root (one directory per paper)")
    parser.add_argument("--runs", required=True, help="output root for per-paper run directories")
    parser.add_argument("--python", default=sys.executable, help="interpreter that runs the chain")
    parser.add_argument("--checkout", default=None, help="working directory for the chain (defaults to cwd)")
    parser.add_argument("--env-file", default=None, help="dotenv file injected into the child environment")
    parser.add_argument("--workers", type=int, default=4, help="concurrent chains")
    parser.add_argument("--timeout", type=float, default=7200.0, help="per-instance timeout in seconds")
    parser.add_argument("--limit", type=int, default=None, help="only the first N instances, in sorted order")
    parser.add_argument("--report", default=None, help="jsonl path (defaults to <runs>/sweep_report.jsonl)")
    parser.add_argument("--resume", action="store_true", help="skip instances that already delivered a deck")
    parser.add_argument("--git", default="git", help="git executable used to record the checkout commit")
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="run the README's offline smoke settings instead of the product configuration",
    )
    for stage, choices in CONFIGURATION_CHOICES.items():
        parser.add_argument(
            "--" + stage.replace("_", "-"),
            dest=stage,
            choices=list(choices),
            default=PRODUCT_CONFIGURATION[stage],
            help="chain stage setting, recorded in the sweep manifest (default: %(default)s)",
        )
    parser.add_argument(
        "--experience-mode",
        choices=EXPERIENCE_MODES,
        default="isolated",
        help="isolated (default) gives every instance its own experience library so instances stay "
        "independent and the benchmark stays reproducible; shared reproduces the accumulating product "
        "behaviour; off disables experience entirely",
    )
    return parser.parse_args(argv)


def load_env(env_file):
    """Return (environment, key names) with the dotenv file merged over os.environ."""
    env = dict(os.environ)
    names = []
    path = pathlib.Path(env_file) if env_file else None
    if path and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
            names.append(key.strip())
    return env, names


def discover_instances(instances_root, limit=None):
    """Return the sorted ids of instance directories that carry a frozen contract."""
    root = pathlib.Path(instances_root)
    if not root.is_dir():
        raise ValueError(f"instances directory does not exist: {root}")
    ids = sorted(child.name for child in root.iterdir() if child.is_dir() and (child / "contract.json").is_file())
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def delivered(run_dir):
    """True once the chain has written an editable deck into this run directory."""
    return (pathlib.Path(run_dir) / "editable_composition.pptx").is_file()


def chain_configuration(args):
    """Return the stage settings this sweep will pass, in a fixed order."""
    if bool(getattr(args, "offline_smoke", False)):
        return dict(OFFLINE_SMOKE_CONFIGURATION)
    return {stage: str(getattr(args, stage, default)) for stage, default in PRODUCT_CONFIGURATION.items()}


def configuration_flags(configuration):
    """Render a configuration as the `--flag value` pairs the chain expects."""
    flags = []
    for stage, value in configuration.items():
        flags += ["--" + stage.replace("_", "-"), str(value)]
    return flags


def configuration_problems(configuration, env):
    """Return why this configuration may not run as asked; empty means it may.

    Asking for a model-backed stage without credentials downgrades the chain to its
    offline path, which is how the 2026-09-28 sweep produced template decks without
    anyone noticing. Refusing up front is the only honest option.
    """
    model_stages = sorted(
        stage for stage, value in configuration.items() if value not in LOCAL_SETTINGS.get(stage, ())
    )
    if not model_stages:
        return []
    missing = [key for key in REQUIRED_CREDENTIALS if not str(env.get(key) or "").strip()]
    if not missing:
        return []
    return [
        "configuration enables model-backed stages (%s) but %s is unset; refusing to run a sweep "
        "that would silently fall back" % (", ".join(model_stages), ", ".join(missing))
    ]


def checkout_commit(args):
    """Return the chain checkout's commit, or an empty string when it is unknown."""
    checkout = getattr(args, "checkout", None)
    if not checkout:
        return ""
    try:
        proc = subprocess.run(
            [str(getattr(args, "git", "git")), "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def write_configuration_manifest(path, configuration, args, env_names):
    """Record what this sweep is about to run, so its numbers stay interpretable."""
    manifest = {
        "summary": "Configuration and harness of one benchmark sweep.",
        "configuration": configuration,
        "offline_smoke": bool(getattr(args, "offline_smoke", False)),
        "python": str(args.python),
        "checkout": str(getattr(args, "checkout", "") or ""),
        "checkout_commit": checkout_commit(args),
        "env_keys": sorted(str(name) for name in env_names),
        "workers": int(args.workers),
        "experience_mode": str(args.experience_mode),
        "instances": str(args.instances),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return manifest


def config_path(args):
    """Where the sweep manifest is written."""
    return pathlib.Path(args.runs) / "sweep_config.json"


def build_command(args, paper_id, pdf_path, run_dir):
    """Build the paper-evolve-editable command for one instance."""
    run_dir = pathlib.Path(run_dir)
    command = [args.python, "-m", "rfs.cli", "paper-evolve-editable", "--paper", str(pdf_path), "--out", str(run_dir)]
    command += configuration_flags(chain_configuration(args))
    if args.experience_mode == "off":
        command.append("--no-experience")
    elif args.experience_mode == "shared":
        command += ["--experience-library", str(pathlib.Path(args.runs) / "experience_library")]
    else:
        command += ["--experience-library", str(run_dir / "experience_library")]
    command.append("--json")
    return command


def run_one(index, paper_id, pdf_path, args, env):
    """Run one instance and return its record; never raises for a failed chain."""
    run_dir = pathlib.Path(args.runs) / paper_id
    record = {
        "paper_id": paper_id,
        "index": index,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configuration": chain_configuration(args),
    }
    if args.resume and delivered(run_dir):
        record.update({"seconds": 0.0, "returncode": 0, "pptx": True, "skipped": "already_delivered"})
        return record
    command = build_command(args, paper_id, pdf_path, run_dir)
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=args.checkout or None,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        record["returncode"] = proc.returncode
        record["stdout_tail"] = proc.stdout[-400:]
        record["stderr_tail"] = proc.stderr[-400:]
    except subprocess.TimeoutExpired:
        record["returncode"] = -1
        record["stderr_tail"] = "timeout after %ss" % args.timeout
    record["seconds"] = round(time.time() - started, 1)
    record["pptx"] = delivered(run_dir)
    record["report_json"] = (run_dir / "paper_evolve_editable_report.json").is_file()
    return record


def report_path(args):
    """Where the per-instance records are appended."""
    return pathlib.Path(args.report) if args.report else pathlib.Path(args.runs) / "sweep_report.jsonl"


def main(argv=None):
    """Run the sweep and return a process exit code."""
    args = parse_args(argv)
    instance_ids = discover_instances(args.instances, args.limit)
    if not instance_ids:
        print("no instances found under", args.instances, file=sys.stderr)
        return 2
    runs_root = pathlib.Path(args.runs)
    runs_root.mkdir(parents=True, exist_ok=True)
    target = report_path(args)
    env, names = load_env(args.env_file)
    configuration = chain_configuration(args)
    problems = configuration_problems(configuration, env)
    if problems:
        for problem in problems:
            print("refusing to start:", problem, file=sys.stderr)
        return 3
    manifest = write_configuration_manifest(config_path(args), configuration, args, names)
    print("env keys loaded:", len(names), flush=True)
    print("configuration:", json.dumps(configuration, ensure_ascii=False), flush=True)
    print("checkout commit:", manifest["checkout_commit"] or "unknown", flush=True)
    print("instances:", len(instance_ids), "workers:", args.workers, flush=True)
    started = time.time()
    written = 0
    with target.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, index, paper_id, pathlib.Path(args.instances) / paper_id / "paper.pdf", args, env): paper_id
            for index, paper_id in enumerate(instance_ids, start=1)
        }
        for future in as_completed(futures):
            record = future.result()
            handle.write(json.dumps(record, ensure_ascii=False) + chr(10))
            handle.flush()
            written += 1
            print(json.dumps({key: record.get(key) for key in ("index", "paper_id", "seconds", "returncode", "pptx", "skipped")}, ensure_ascii=False), flush=True)
    decks = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("pptx"))
    print("DONE records=%d deck_written=%d wall_seconds=%.1f" % (written, decks, time.time() - started), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
