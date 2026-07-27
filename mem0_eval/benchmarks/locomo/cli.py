from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .data import load_conversations, sample_questions
from .generator import AnswerGenerator, ConversationSummarizer
from .protocol import run_benchmark, summarize_evaluations


def add_common_arguments(parser: argparse.ArgumentParser, root: Path) -> None:
    parser.add_argument("--dataset", type=Path, default=root / "data/locomo10.json")
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--questions-per-category", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
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
            pair_count = len(
                conversation.incremental_exchanges(len(sessions))
            )
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
                    "speakers": list(conversation.speakers),
                    "sessions": sessions,
                    "message_pair_count": pair_count,
                    "memory_update_count": pair_count
                    * len(conversation.speakers),
                    "summary_update_count": math.ceil(pair_count / 5),
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
        ConversationSummarizer(),
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
        "schema_version": 4,
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
            "ingestion_unit": "one chronological user-assistant message pair",
            "speaker_memory_scopes": "separate scope for each LoCoMo speaker",
            "recent_context_messages": 10,
            "conversation_summary": (
                "key knowledge organized by person and shared timeline; "
                "refreshed every 5 message pairs"
            ),
            "summary_output_token_limit": 800,
            "session_boundaries": (
                "rolling context and summary continue across session boundaries"
            ),
            "paper_alignment": (
                "Mem0 paper section 2.1 extraction context and appendix answer prompt"
            ),
            "known_paper_differences": [
                "DeepSeek replaces GPT-4o-mini",
                "BAAI/bge-small-en-v1.5 replaces the paper model stack",
                "summary refresh is synchronous every 5 pairs; the paper does not publish its cadence",
                "text backend uses mem0ai 2.0.14 ADD-only internals",
                "graph backend uses historical mem0ai 0.1.45",
            ],
            "top_k": args.top_k,
            "answer_model": "same DeepSeek model for both backends",
            "answer_prompt": "Mem0 paper appendix results-generation prompt",
            "answer_metrics": [
                "official LoCoMo token F1",
                "Mem0 paper appendix LLM-as-a-judge prompt",
            ],
            "judge_order": "run after token F1 for each generated answer",
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
