from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

from mem0_eval.benchmarks.eval_stats import (
    bootstrap_mean_interval,
    paired_difference_interval,
)
from mem0_eval.benchmarks.memory_changes.metrics import percentile

from .data import CATEGORY_NAMES, LoCoMoConversation, LoCoMoQuestion, sample_questions
from .metrics import answer_token_recall, official_locomo_f1


class MemoryAdapter(Protocol):
    def add(self, statement: str, *, user_id: str) -> Any: ...
    def search(self, query: str, *, user_id: str) -> Any: ...
    def delete_all(self, *, user_id: str) -> Any: ...


class Generator(Protocol):
    def answer(self, *, question: str, category: int, retrieval: Any) -> str: ...


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, round((time.perf_counter() - started) * 1000, 3)


def run_conversation(
    adapter: MemoryAdapter,
    generator: Generator,
    conversation: LoCoMoConversation,
    *,
    user_id: str,
    session_count: int,
    questions_per_category: int,
    categories: tuple[int, ...],
    seed: int,
    include_no_memory_control: bool,
) -> dict[str, Any]:
    sessions = conversation.session_keys(session_count)
    questions = sample_questions(
        conversation,
        session_count=len(sessions),
        questions_per_category=questions_per_category,
        categories=categories,
        seed=seed,
    )
    ingestion: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    status = "completed"
    error = None
    try:
        for key in sessions:
            result, latency = _timed(
                lambda session_key=key: adapter.add(
                    conversation.session_text(session_key), user_id=user_id
                )
            )
            ingestion.append({"session": key, "latency_ms": latency, "result": result})
        for question in questions:
            try:
                retrieval, retrieval_ms = _timed(
                    lambda item=question: adapter.search(
                        item.question, user_id=user_id
                    )
                )
                answer, answer_ms = _timed(
                    lambda item=question, memory=retrieval: generator.answer(
                        question=item.question,
                        category=item.category,
                        retrieval=memory,
                    )
                )
                control_answer = (
                    generator.answer(
                        question=question.question,
                        category=question.category,
                        retrieval=[],
                    )
                    if include_no_memory_control
                    else None
                )
                f1 = official_locomo_f1(answer, question.answer, question.category)
                control_f1 = (
                    official_locomo_f1(
                        control_answer, question.answer, question.category
                    )
                    if control_answer is not None
                    else None
                )
                evaluations.append(
                    {
                        "status": "completed",
                        "question_id": question.question_id,
                        "conversation_index": conversation.conversation_index,
                        "category": question.category,
                        "category_name": CATEGORY_NAMES[question.category],
                        "question": question.question,
                        "expected_answer": question.answer,
                        "evidence": list(question.evidence),
                        "retrieval": retrieval,
                        "retrieval_answer_token_recall": answer_token_recall(
                            json.dumps(
                                retrieval, ensure_ascii=False, default=str
                            ),
                            question.answer,
                        ),
                        "generated_answer": answer,
                        "official_f1": round(f1, 4),
                        "no_memory_answer": control_answer,
                        "no_memory_official_f1": (
                            round(control_f1, 4)
                            if control_f1 is not None
                            else None
                        ),
                        "memory_f1_lift": (
                            round(f1 - control_f1, 4)
                            if control_f1 is not None
                            else None
                        ),
                        "retrieval_latency_ms": retrieval_ms,
                        "answer_latency_ms": answer_ms,
                    }
                )
            except Exception as exc:
                evaluations.append(
                    {
                        "status": "failed",
                        "question_id": question.question_id,
                        "conversation_index": conversation.conversation_index,
                        "category": question.category,
                        "category_name": CATEGORY_NAMES[question.category],
                        "question": question.question,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            adapter.delete_all(user_id=user_id)
        except Exception:
            pass
    return {
        "status": status,
        "error": error,
        "conversation_index": conversation.conversation_index,
        "sample_id": conversation.sample_id,
        "sessions": sessions,
        "ingestion": ingestion,
        "evaluations": evaluations,
    }


def run_benchmark(
    adapter: MemoryAdapter,
    generator: Generator,
    conversations: list[LoCoMoConversation],
    *,
    conversation_limit: int,
    user_id_prefix: str,
    session_count: int,
    questions_per_category: int,
    categories: tuple[int, ...],
    seed: int,
    include_no_memory_control: bool,
) -> list[dict[str, Any]]:
    return [
        run_conversation(
            adapter,
            generator,
            conversation,
            user_id=f"{user_id_prefix}_{conversation.conversation_index}",
            session_count=session_count,
            questions_per_category=questions_per_category,
            categories=categories,
            seed=seed,
            include_no_memory_control=include_no_memory_control,
        )
        for conversation in conversations[:conversation_limit]
    ]


def _p(values: list[float], probability: float) -> float | None:
    value = percentile(values, probability)
    return round(value, 3) if value is not None else None


def summarize_evaluations(runs: list[dict[str, Any]]) -> dict[str, Any]:
    evaluations = [item for run in runs for item in run["evaluations"]]
    completed = [item for item in evaluations if item["status"] == "completed"]
    controlled = [
        item for item in completed if item["no_memory_official_f1"] is not None
    ]
    by_category: dict[str, list[float]] = defaultdict(list)
    for item in completed:
        by_category[item["category_name"]].append(item["official_f1"])
    f1_values = [item["official_f1"] for item in completed]
    control_values = [item["no_memory_official_f1"] for item in controlled]
    paired_memory = [item["official_f1"] for item in controlled]
    retrieval_latencies = [item["retrieval_latency_ms"] for item in completed]
    ingestion_latencies = [
        item["latency_ms"] for run in runs for item in run["ingestion"]
    ]

    def nonempty(value: Any) -> bool:
        if isinstance(value, dict):
            value = value.get("results", [])
        return bool(value)

    return {
        "conversation_count": len(runs),
        "completed_conversation_count": sum(run["status"] == "completed" for run in runs),
        "question_count": len(evaluations),
        "completed_question_count": len(completed),
        "failed_question_count": len(evaluations) - len(completed),
        "mean_official_f1": (
            round(sum(f1_values) / len(f1_values), 4) if f1_values else None
        ),
        "mean_official_f1_95ci": bootstrap_mean_interval(f1_values),
        "mean_no_memory_f1": (
            round(sum(control_values) / len(control_values), 4)
            if control_values
            else None
        ),
        "mean_memory_f1_lift": (
            round(
                sum(a - b for a, b in zip(paired_memory, control_values, strict=True))
                / len(controlled),
                4,
            )
            if controlled
            else None
        ),
        "mean_memory_f1_lift_95ci": paired_difference_interval(
            paired_memory, control_values
        ),
        "mean_retrieval_answer_token_recall": (
            round(
                sum(
                    item["retrieval_answer_token_recall"]
                    for item in completed
                    if item["retrieval_answer_token_recall"] is not None
                )
                / sum(
                    item["retrieval_answer_token_recall"] is not None
                    for item in completed
                ),
                4,
            )
            if any(
                item["retrieval_answer_token_recall"] is not None
                for item in completed
            )
            else None
        ),
        "nonempty_retrieval_rate": (
            sum(nonempty(item["retrieval"]) for item in completed) / len(completed)
            if completed
            else None
        ),
        "answer_abstention_rate": (
            sum(
                item["generated_answer"]
                .casefold()
                .startswith("not mentioned in the retrieved memory")
                for item in completed
            )
            / len(completed)
            if completed
            else None
        ),
        "f1_by_category": {
            label: {
                "count": len(values),
                "mean": round(sum(values) / len(values), 4),
                "95ci": bootstrap_mean_interval(values),
            }
            for label, values in sorted(by_category.items())
        },
        "ingestion_latency_ms": {
            "p50": _p(ingestion_latencies, 0.50),
            "p95": _p(ingestion_latencies, 0.95),
        },
        "retrieval_latency_ms": {
            "p50": _p(retrieval_latencies, 0.50),
            "p95": _p(retrieval_latencies, 0.95),
        },
    }
