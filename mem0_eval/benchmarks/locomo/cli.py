from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .data import load_conversations, sample_questions
from .generator import AnswerGenerator
from .protocol import run_benchmark, summarize_evaluations


def add_common_arguments(parser: argparse.ArgumentParser, root: Path) -> None:
    parser.add_argument("--dataset", type=Path, default=root / "data/locomo10.json")
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--questions-per-category", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-no-memory-control", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=root / "results/locomo")


def execute(
    *,
    args: argparse.Namespace,
    backend_name: str,
    adapter_builder: Any,
    extra_environment: dict[str, Any] | None = None,
) -> int:
    conversations = load_conversations(args.dataset)
    categories = (1, 2, 3, 4)
    if args.dry_run:
        preview = []
        for conversation in conversations[: args.conversations]:
            sessions = conversation.session_keys(args.sessions)
            questions = sample_questions(
                conversation,
                session_count=len(sessions),
                questions_per_category=args.questions_per_category,
                categories=categories,
                seed=args.seed,
            )
            preview.append(
                {
                    "conversation_index": conversation.conversation_index,
                    "sessions": sessions,
                    "questions": [
                        {
                            "question_id": item.question_id,
                            "category": item.category,
                            "evidence": item.evidence,
                        }
                        for item in questions
                    ],
                }
            )
        print(json.dumps(preview, indent=2))
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid4().hex[:8]
    adapter = adapter_builder(args, run_id, token)
    runs = run_benchmark(
        adapter,
        AnswerGenerator(),
        conversations,
        conversation_limit=args.conversations,
        user_id_prefix=f"locomo_{backend_name}_{run_id}_{token}",
        session_count=args.sessions,
        questions_per_category=args.questions_per_category,
        categories=categories,
        seed=args.seed,
        include_no_memory_control=not args.skip_no_memory_control,
    )
    report = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "dataset": str(args.dataset),
        "sample": {
            "seed": args.seed,
            "conversation_limit": args.conversations,
            "session_limit_per_conversation": args.sessions,
            "questions_per_category_per_conversation": args.questions_per_category,
            "categories": list(categories),
        },
        "protocol": {
            "ingestion_unit": "one complete dated session",
            "top_k": args.top_k,
            "answer_model": "same DeepSeek model for both backends",
            "answer_metric": "official LoCoMo token F1",
            "paired_no_memory_control": not args.skip_no_memory_control,
            "adversarial_category_5": "excluded to match Mem0 paper",
            "confidence_intervals": (
                "95% question-level percentile bootstrap; Wilson for Bernoulli metrics"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "mem0ai": version("mem0ai"),
            **(extra_environment or {}),
        },
        "summary": summarize_evaluations(runs),
        "conversations": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{run_id}_{token}_{backend_name}.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0 if report["summary"]["completed_question_count"] else 1
