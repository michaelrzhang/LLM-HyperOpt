# ABOUTME: Headless CLI for inspecting studies (`llm-hyperopt report|best|list`) so a run's
# ABOUTME: state is queryable in CI or a terminal without an agent in the loop.

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .study import Study


def _add_storage_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage-root",
        default=".llm_hyperopt",
        help="Directory where studies are stored (default: .llm_hyperopt)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-hyperopt",
        description="Inspect LLM-HyperOpt studies from the command line.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="Print a study's progress summary")
    p_report.add_argument("name")
    _add_storage_arg(p_report)

    p_best = sub.add_parser("best", help="Print the best trial as JSON")
    p_best.add_argument("name")
    _add_storage_arg(p_best)

    p_list = sub.add_parser("list", help="List available studies")
    _add_storage_arg(p_list)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        root = Path(args.storage_root)
        if not root.exists():
            print(f"No studies found under {root}")
            return 0
        names = sorted(p.name for p in root.iterdir() if (p / "study.json").exists())
        if not names:
            print(f"No studies found under {root}")
        for name in names:
            print(name)
        return 0

    try:
        study = Study.load(args.name, storage_root=args.storage_root)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.command == "report":
        print(study.report())
        return 0

    if args.command == "best":
        best = study.best()
        if best is None:
            print(json.dumps({"metric": None, "params": None}))
        else:
            print(json.dumps({"metric": best.metric, "params": best.params, "trial": best.id}))
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
