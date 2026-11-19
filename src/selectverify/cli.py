"""Portable command-line entry points for the paper experiments."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .benchmark import ASSETS, load
from .io import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="SelectVerify: budgeted failure attribution for tool-using agents.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("check-data", "Validate the fixed manifest and show dev/test counts"),
                            ("run", "Run SV, SV-Lite, AAO, Binary Search or matched controls"),
                            ("collect", "Collect exhaustive signals for dev search and test ablations"),
                            ("screen", "Select configurations on dev only (784 candidates)"),
                            ("evaluate", "Evaluate frozen configurations using cached test signals")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--data-root", type=Path, required=True, help="Who&When Pro directory containing data/text.jsonl and taxonomy.yaml")
        sub.add_argument("--benchmark-source", type=Path, help="Optional checkout containing whowhen_eval; otherwise install the upstream package")
        sub.add_argument("--manifest", type=Path, default=ASSETS / "pro_manifest.json")
        if name in ("run", "collect"):
            sub.add_argument("--model", required=True, help="Explicit local checkpoint path or Hugging Face model ID; no implicit model choice")
            sub.add_argument("--revision", help="Fixed Hugging Face model revision when using a remote model ID")
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--split", choices=("dev", "test", "all"), default="dev")
            sub.add_argument("--per-framework", type=int, default=0, help="Deterministic smoke-test subset per framework; zero means all")
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
            sub.add_argument("--local-batch", type=int, default=3)
        if name == "run":
            sub.add_argument("--methods", default="lean,best_dev,aao", help="Comma-separated: lean,best_dev,aao,binary_search,short,reason")
            sub.add_argument("--config-dir", type=Path, default=ASSETS)
            sub.add_argument("--binary-source", type=Path, help="Upstream Automated_FA/Lib/utils.py; required only for binary_search")
            sub.add_argument("--paired-source", type=Path, help="Prior best_dev JSONL; required for identical-candidate short/reason controls")
        if name in ("screen", "evaluate"):
            sub.add_argument("--cache-dir", type=Path, required=True, help="signals*.jsonl and all_at_once_4b*.jsonl")
            sub.add_argument("--out-dir", type=Path, required=True, help="Selected config JSONs and cached evaluation outputs")
    report_parser = commands.add_parser("report", help="Recompute metrics from JSONL without a GPU or benchmark data")
    source_group = report_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path)
    source_group.add_argument("--paper", action="store_true", help="Summarize bundled archived paper scores")
    report_parser.add_argument("--output", type=Path)
    report_parser.add_argument("--methods", help="Expected comma-separated methods; incomplete populations are rejected")
    args = parser.parse_args(argv)
    if getattr(args, "benchmark_source", None):
        source = args.benchmark_source.resolve()
        if not (source / "whowhen_eval/__init__.py").is_file():
            parser.error("--benchmark-source must contain whowhen_eval/__init__.py")
        sys.path.insert(0, str(source))
    if args.command == "report":
        from .report import summarize
        source = ASSETS / "paper_scores.jsonl" if args.paper else args.input
        result = summarize(source, args.methods.split(",") if args.methods else None)
        if args.paper:
            result["source"] = "Archived paper measurements"
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command in ("screen", "evaluate"):
        from . import cached
        args.out_dir.mkdir(parents=True, exist_ok=True)
        getattr(cached, args.command)(args.out_dir, args.cache_dir, args.data_root, args.manifest)
    elif args.command == "check-data":
        samples = load(args.data_root, args.manifest)
        print(json.dumps({"n": len(samples), "splits": dict(Counter(s.split for s in samples)),
                          "frameworks": dict(Counter(s.framework for s in samples)),
                          "inference_fields": list(samples[0].trace.__dataclass_fields__) if samples else []}, indent=2))
    else:
        if args.local_batch < 1 or args.per_framework < 0:
            parser.error("--local-batch must be positive; --per-framework must be nonnegative")
        from .experiments import run
        run(args)


if __name__ == "__main__":
    main()
