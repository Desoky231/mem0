from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .data import load_cases, sample_cases
from .protocol import run_benchmark, summarize_evaluations
from .selector import MCQSelector


def add_common_arguments(parser: argparse.ArgumentParser, root: Path) -> None:
    parser.add_argument(
        "--dataset", type=Path, default=root / "data/personamem_v2.json"
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-sensitive", action="store_true")
    parser.add_argument("--include-multimodal", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer-selection calls; report retrieval diagnostics only.",
    )
    parser.add_argument(
        "--skip-no-memory-control",
        action="store_true",
        help="Do not run the paired answer-selection control with empty memory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the deterministic sample without using a backend.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/personamem_v2"
    )


def execute(
    *,
    args: argparse.Namespace,
    root: Path,
    backend_name: str,
    adapter_builder: Any,
    extra_environment: dict[str, Any] | None = None,
) -> int:
    cases = sample_cases(
        load_cases(args.dataset),
        limit=args.limit,
        seed=args.seed,
        include_sensitive=args.include_sensitive,
        include_multimodal=args.include_multimodal,
    )
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "case_id": case.case_id,
                        "preference_type": case.preference_type,
                        "updated": case.updated,
                        "who": case.who,
                        "sensitive": case.sensitive,
                    }
                    for case in cases
                ],
                indent=2,
            )
        )
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid4().hex[:8]
    adapter = adapter_builder(args, run_id, token)
    selector = None if args.retrieval_only else MCQSelector()
    evaluations = run_benchmark(
        adapter,
        cases,
        user_id_prefix=f"personamem_{backend_name}_{run_id}_{token}",
        seed=args.seed,
        selector=selector,
        include_no_memory_control=not args.skip_no_memory_control,
    )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "dataset": str(args.dataset),
        "sample": {
            "seed": args.seed,
            "requested_limit": args.limit,
            "sensitive_included": args.include_sensitive,
            "multimodal_included": args.include_multimodal,
        },
        "protocol": {
            "memory_input": "related_conversation_snippet only",
            "forbidden_memory_inputs": [
                "preference",
                "prev_pref",
                "correct_answer",
                "incorrect_answers",
                "short_persona",
                "expanded_persona",
            ],
            "top_k": args.top_k,
            "answer_metric": "exact MCQ accuracy" if selector else None,
            "paired_no_memory_control": (
                selector is not None and not args.skip_no_memory_control
            ),
            "forget_metric": (
                "Exact previous-preference phrase present after the forget request"
            ),
            "isolation": "one unique memory user_id per dataset row",
        },
        "environment": {
            "python": platform.python_version(),
            "mem0ai": version("mem0ai"),
            **(extra_environment or {}),
        },
        "summary": summarize_evaluations(evaluations),
        "evaluations": evaluations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{run_id}_{token}_{backend_name}.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0 if report["summary"]["completed_count"] else 1
