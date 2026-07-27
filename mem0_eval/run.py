#!/usr/bin/env python3
"""Single command interface for all memory backends and benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from mem0_eval.benchmarks.locomo.cli import (
    add_common_arguments as add_locomo_arguments,
)
from mem0_eval.benchmarks.locomo.cli import execute as execute_locomo
from mem0_eval.benchmarks.memory_changes.case import MemoryChangeCase
from mem0_eval.benchmarks.memory_changes.protocol import (
    run_case,
    summarize_runs,
)
from mem0_eval.benchmarks.personamem.cli import (
    add_common_arguments as add_personamem_arguments,
)
from mem0_eval.benchmarks.personamem.cli import execute as execute_personamem


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GRAPH_MEM0_VERSION = "0.1.45"
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    "main/data/locomo10.json"
)


def _add_backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("text", "graph"),
        default="text",
        help="Memory implementation to evaluate (default: text).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Mem0 evaluations.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "download",
        help="Download the LoCoMo and PersonaMem-v2 datasets.",
    )
    commands.add_parser(
        "check-graph",
        help="Check the graph environment and Neo4j connection.",
    )

    locomo = commands.add_parser(
        "locomo",
        help="Evaluate long-term conversation recall.",
    )
    add_locomo_arguments(locomo, ROOT)
    _add_backend_argument(locomo)
    locomo.add_argument("--threshold", type=float, default=0.0)

    personamem = commands.add_parser(
        "personamem",
        help="Evaluate whether memory improves personalized answers.",
    )
    add_personamem_arguments(personamem, ROOT)
    _add_backend_argument(personamem)
    personamem.add_argument("--threshold", type=float, default=0.0)

    changes = commands.add_parser(
        "memory-changes",
        help="Check whether old facts remain after an update or deletion.",
    )
    _add_backend_argument(changes)
    changes.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "mem0_eval/benchmarks/memory_changes/cases.json",
    )
    changes.add_argument("--limit", type=int)
    changes.add_argument("--top-k", type=int, default=5)
    changes.add_argument("--threshold", type=float, default=0.0)
    changes.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/memory_changes",
    )
    return parser


def download_datasets() -> int:
    import requests
    from datasets import load_dataset

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    locomo_path = data_dir / "locomo10.json"
    if not locomo_path.exists():
        response = requests.get(LOCOMO_URL, timeout=60)
        response.raise_for_status()
        locomo_path.write_text(response.text, encoding="utf-8")
        print(f"Downloaded {locomo_path}")
    else:
        print(f"Already present: {locomo_path}")

    personamem_path = data_dir / "personamem_v2.json"
    if not personamem_path.exists():
        dataset = load_dataset("bowen-upenn/PersonaMem-v2")
        split = dataset[next(iter(dataset))]
        rows = [dict(row) for row in split]
        personamem_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Downloaded {personamem_path}")
    else:
        print(f"Already present: {personamem_path}")
    return 0


def _backend_name(backend: str) -> str:
    return (
        "mem0_text_v2.0.14"
        if backend == "text"
        else "mem0_graph_v0.1.45"
    )


def _graph_environment() -> dict[str, Any]:
    installed = version("mem0ai")
    if installed != GRAPH_MEM0_VERSION:
        raise RuntimeError(
            f"Graph memory requires mem0ai {GRAPH_MEM0_VERSION}; found "
            f"{installed}. Run with --project mem0_eval/backends/graph."
        )
    from mem0_eval.backends.graph.adapter import verify_neo4j

    return {
        "neo4j": verify_neo4j(),
        "embedding_model": EMBEDDING_MODEL,
    }


def _build_adapter(
    *,
    backend: str,
    state_dir: Path,
    top_k: int,
    threshold: float,
) -> Any:
    if backend == "text":
        from mem0_eval.backends.text.adapter import build_text_memory

        return build_text_memory(
            state_dir=state_dir,
            top_k=top_k,
            threshold=threshold,
        )

    from mem0_eval.backends.graph.adapter import build_graph_adapter

    return build_graph_adapter(top_k=top_k)


def run_locomo(args: argparse.Namespace) -> int:
    details: dict[str, Any] = {"embedding_model": EMBEDDING_MODEL}
    if args.backend == "graph" and not args.dry_run:
        details = _graph_environment()
    return execute_locomo(
        args=args,
        backend_name=_backend_name(args.backend),
        adapter_builder=lambda parsed, run_id, token: _build_adapter(
            backend=parsed.backend,
            state_dir=(
                parsed.output_dir
                / "state"
                / f"{run_id}_{token}_{parsed.backend}"
            ),
            top_k=parsed.top_k,
            threshold=parsed.threshold,
        ),
        extra_environment=details,
    )


def run_personamem(args: argparse.Namespace) -> int:
    details: dict[str, Any] = {"embedding_model": EMBEDDING_MODEL}
    if args.backend == "graph" and not args.dry_run:
        details = _graph_environment()
    return execute_personamem(
        args=args,
        root=ROOT,
        backend_name=_backend_name(args.backend),
        adapter_builder=lambda parsed, run_id, token: _build_adapter(
            backend=parsed.backend,
            state_dir=(
                parsed.output_dir
                / "state"
                / f"{run_id}_{token}_{parsed.backend}"
            ),
            top_k=parsed.top_k,
            threshold=parsed.threshold,
        ),
        extra_environment=details,
    )


def run_memory_changes(args: argparse.Namespace) -> int:
    raw_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [MemoryChangeCase.from_dict(item) for item in raw_cases]
    if args.limit is not None:
        cases = cases[: args.limit]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid4().hex[:8]
    adapter = _build_adapter(
        backend=args.backend,
        state_dir=(
            args.output_dir / "state" / f"{run_id}_{token}_{args.backend}"
        ),
        top_k=args.top_k,
        threshold=args.threshold,
    )
    runs = [
        run_case(
            adapter,
            case,
            user_id=(
                f"memory_change_{run_id}_{token}_{args.backend}_{case.case_id}"
            ),
        )
        for case in cases
    ]
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "mem0ai": version("mem0ai"),
        "top_k": args.top_k,
    }
    if args.backend == "graph":
        details.update(_graph_environment())
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": _backend_name(args.backend),
        "environment": details,
        "summary": summarize_runs(runs),
        "cases": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"{run_id}_{token}_{args.backend}.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(ROOT / ".env")
    if args.command == "download":
        return download_datasets()
    if args.command == "check-graph":
        print(json.dumps(_graph_environment(), indent=2))
        return 0
    if args.command == "locomo":
        return run_locomo(args)
    if args.command == "personamem":
        return run_personamem(args)
    if args.command == "memory-changes":
        return run_memory_changes(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
