"""Prepare and collect the LiveFigure baseline for EditableFigureBench.

Two subcommands, both resumable:

- `describe`: turn each frozen instance's paper into the textual description that
  LiveFigure consumes, and write the JSONL their `App/batch_runner.py` reads.
- `collect`: pull the PowerPoint that LiveFigure produced back into this
  benchmark's system layout so `rfs benchmark run` can score it.

LiveFigure itself is not vendored: the authors' repository ships no LICENSE, so it
is run from a pinned checkout (see `baselines/livefigure/baseline.json`) and only
its outputs enter this repository's data directories. The paper-to-description
conversion is ours, fixed and versioned, and is disclosed as part of the baseline.
It reads the paper only: no gold contract is ever passed to the model.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

DEFAULT_PROMPT = pathlib.Path(__file__).resolve().parents[1] / "benchmarks/editable-figure-bench/baselines/livefigure/prompts/describe-v001.md"
DESCRIPTION_KEYS = ("caption", "description")
PAPER_MARKER = "{paper_text}"


def parse_args(argv=None):
    """Parse the two subcommands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="action", required=True)

    describe = subparsers.add_parser("describe", help="paper -> description -> LiveFigure dataset JSONL")
    describe.add_argument("--instances", required=True)
    describe.add_argument("--out", required=True)
    describe.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    describe.add_argument("--model", default=None, help="fixed model id; defaults to RFS_BASELINE_MODEL or the pipeline VLM")
    describe.add_argument("--chars", type=int, default=12000, help="paper character budget handed to the model")
    describe.add_argument("--limit", type=int, default=None)
    describe.add_argument("--resume", action="store_true")
    describe.add_argument("--json", action="store_true")

    collect = subparsers.add_parser("collect", help="LiveFigure output -> benchmark system layout")
    collect.add_argument("--livefigure-root", required=True, help="the OUTPUT_ROOT given to batch_runner")
    collect.add_argument("--instances", required=True)
    collect.add_argument("--systems", required=True, help="writes <systems>/<system_id>/<paper_id>/")
    collect.add_argument("--system-id", default="livefigure")
    collect.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def discover_instances(instances_root, limit=None):
    """Return the sorted ids of instance directories carrying a frozen contract."""
    root = pathlib.Path(instances_root)
    if not root.is_dir():
        raise ValueError("instances directory does not exist: " + str(root))
    ids = sorted(child.name for child in root.iterdir() if child.is_dir() and (child / "contract.json").is_file())
    return ids[: int(limit)] if limit else ids


def load_prompt(path):
    """Read the versioned prompt template and refuse one without the paper slot."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if PAPER_MARKER not in text:
        raise ValueError("prompt template has no " + PAPER_MARKER + " slot: " + str(path))
    return text


def build_prompt(template, paper_text):
    """Fill the template's paper slot."""
    return template.replace(PAPER_MARKER, str(paper_text))


def prompt_fingerprint(template):
    """Return the sha256 of the template, so every response names its exact prompt."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def paper_text(pdf_path, chars):
    """Extract the paper's own text with the repository's canonical reader."""
    from rfs.analysis.paper.document import parse_paper, paper_markdown

    parsed = parse_paper(pdf_path, max_chars=max(4000, int(chars) * 3))
    text = paper_markdown(parsed)
    return text[: int(chars)]


def validate_description(payload):
    """Return the caption and description, refusing an answer that is missing either."""
    if not isinstance(payload, dict):
        raise ValueError("model answer is not an object")
    missing = [key for key in DESCRIPTION_KEYS if not str(payload.get(key) or "").strip()]
    if missing:
        raise ValueError("model answer is missing: " + ", ".join(missing))
    return str(payload["caption"]).strip(), str(payload["description"]).strip()


def dataset_line(paper_id, caption, description):
    """One line of the JSONL that `App/batch_runner.py` reads.

    `paper_name` carries the instance id and `figure_label` is fixed, so the case
    directory LiveFigure creates (`<paper_name>_<figure_label>`) maps back to the
    instance without guessing.
    """
    return {"paper_name": paper_id, "figure_label": "Fig1", "caption": caption, "description": description}


def case_directory(livefigure_root, paper_id):
    """The directory LiveFigure creates for one instance."""
    return pathlib.Path(livefigure_root) / (paper_id + "_Fig1")


def select_deck(case_dir):
    """Return the newest generated deck under one LiveFigure case directory."""
    directory = pathlib.Path(case_dir)
    if not directory.is_dir():
        return None
    decks = [path for path in directory.rglob("*.pptx") if "code_iter" in path.name]
    if not decks:
        return None
    return max(decks, key=lambda path: (path.stat().st_mtime, str(path)))


def select_preview(case_dir, deck):
    """Return a PNG to use as the preview, preferring the newest next to the deck."""
    directory = pathlib.Path(case_dir)
    if not directory.is_dir():
        return None
    images = [path for path in directory.rglob("*.png")]
    if not images:
        return None
    if deck is not None:
        nearby = [path for path in images if path.parent == pathlib.Path(deck).parent]
        if nearby:
            images = nearby
    return max(images, key=lambda path: (path.stat().st_mtime, str(path)))


def describe_one(paper_id, instance_dir, args, template, fingerprint):
    """Describe one paper and return its response record, or raise."""
    from rfs.providers import call_vlm_json, resolve_vlm_model

    model = args.model or os.getenv("RFS_BASELINE_MODEL") or resolve_vlm_model()
    text = paper_text(instance_dir / "paper.pdf", args.chars)
    prompt = build_prompt(template, text)
    metadata = {}
    answer = call_vlm_json(prompt, [], model=model, retries=1, call_metadata=metadata)
    caption, description = validate_description(answer)
    return {
        "paper_id": paper_id,
        "model": model,
        "prompt_version": 1,
        "prompt_sha256": fingerprint,
        "paper_sha256": hashlib.sha256((instance_dir / "paper.pdf").read_bytes()).hexdigest(),
        "paper_chars": len(text),
        "caption": caption,
        "description": description,
        "usage": metadata,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_describe(args):
    """Describe every instance and rewrite the dataset JSONL."""
    template = load_prompt(args.prompt)
    fingerprint = prompt_fingerprint(template)
    out = pathlib.Path(args.out)
    responses = out / "responses"
    responses.mkdir(parents=True, exist_ok=True)
    instance_ids = discover_instances(args.instances, args.limit)
    if not instance_ids:
        raise ValueError("no instances found under " + str(args.instances))
    records = []
    failures = []
    for paper_id in instance_ids:
        target = responses / (paper_id + ".json")
        if args.resume and target.is_file():
            records.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        try:
            record = describe_one(paper_id, pathlib.Path(args.instances) / paper_id, args, template, fingerprint)
        except Exception as exc:
            failures.append({"paper_id": paper_id, "error": str(exc)[:300]})
            print("FAILED " + paper_id + ": " + str(exc)[:200], flush=True)
            continue
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records.append(record)
        print("described " + paper_id, flush=True)
    lines = [dataset_line(record["paper_id"], record["caption"], record["description"]) for record in records]
    dataset = out / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    (out / "describe_report.json").write_text(
        json.dumps({"described": len(records), "failed": failures, "prompt_sha256": fingerprint, "dataset": str(dataset)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps({"described": len(records), "failed": len(failures), "dataset": str(dataset)}, ensure_ascii=False))
    return 0 if not failures else 1


def run_collect(args):
    """Copy the decks LiveFigure produced into the benchmark's system layout."""
    systems = pathlib.Path(args.systems) / args.system_id
    instance_ids = discover_instances(args.instances)
    collected = []
    missing = []
    for paper_id in instance_ids:
        case_dir = case_directory(args.livefigure_root, paper_id)
        deck = select_deck(case_dir)
        if deck is None:
            missing.append(paper_id)
            continue
        target = systems / paper_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "editable_composition.pptx").write_bytes(deck.read_bytes())
        preview = select_preview(case_dir, deck)
        if preview is not None:
            (target / "preview.png").write_bytes(preview.read_bytes())
        collected.append({"paper_id": paper_id, "deck": str(deck), "preview": str(preview) if preview else None})
    report = {"system_id": args.system_id, "systems_dir": str(systems), "collected": len(collected), "missing": missing, "instances": collected}
    (systems.parent / (args.system_id + "_collect_report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps({"collected": len(collected), "missing": len(missing)}, ensure_ascii=False))
    return 0


def main(argv=None):
    """Dispatch one subcommand."""
    args = parse_args(argv)
    if args.action == "describe":
        return run_describe(args)
    return run_collect(args)


if __name__ == "__main__":
    sys.exit(main())
