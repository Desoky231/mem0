from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any, Protocol

from mem0_eval.benchmarks.eval_stats import (
    bootstrap_mean_interval,
    paired_difference_interval,
    wilson_interval,
)
from mem0_eval.benchmarks.memory_changes.metrics import percentile

from .data import (
    CATEGORY_NAMES,
    LoCoMoConversation,
    LoCoMoExchange,
    LoCoMoQuestion,
    sample_questions,
)
from .metrics import answer_token_recall, official_locomo_f1


SUMMARY_PAIR_INTERVAL = 5


class MemoryAdapter(Protocol):
    def add(self, statement: str, *, user_id: str) -> Any: ...
    def add_exchange(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        speaker: str,
        session: str,
        session_date: str,
        conversation_summary: str,
        recent_messages: list[str],
    ) -> Any: ...
    def search(self, query: str, *, user_id: str) -> Any: ...
    def delete_all(self, *, user_id: str) -> Any: ...


class Generator(Protocol):
    def answer(self, *, question: str, category: int, retrieval: Any) -> str: ...
    def judge(
        self,
        *,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> str: ...


class Summarizer(Protocol):
    def update(
        self,
        *,
        previous_summary: str,
        batch_index: int,
        messages: list[str],
    ) -> str: ...


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, round((time.perf_counter() - started) * 1000, 3)


def _speaker_ids(
    conversation: LoCoMoConversation, user_id: str
) -> dict[str, str]:
    return {
        speaker: f"{user_id}_speaker_{index}"
        for index, speaker in enumerate(conversation.speakers, start=1)
    }


def _messages_for_speaker(
    exchange: LoCoMoExchange, target_speaker: str
) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if message.speaker == target_speaker else "assistant",
            "content": message.formatted(),
        }
        for message in exchange.messages
    ]


def _search_both_speakers(
    adapter: MemoryAdapter,
    question: str,
    *,
    speaker_ids: dict[str, str],
) -> dict[str, Any]:
    return {
        "speakers": [
            {
                "speaker": speaker,
                "user_id": speaker_id,
                "memories": adapter.search(question, user_id=speaker_id),
            }
            for speaker, speaker_id in speaker_ids.items()
        ]
    }


def _empty_retrieval(speaker_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "speakers": [
            {
                "speaker": speaker,
                "user_id": speaker_id,
                "memories": [],
            }
            for speaker, speaker_id in speaker_ids.items()
        ]
    }


def run_conversation(
    adapter: MemoryAdapter,
    generator: Generator,
    summarizer: Summarizer,
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
    summaries: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    speaker_ids = _speaker_ids(conversation, user_id)
    recent_messages: deque[str] = deque(maxlen=10)
    pending_summary_messages: list[str] = []
    pending_summary_exchange_ids: list[str] = []
    summary_batch_index = 0
    conversation_summary = ""
    status = "completed"
    error = None
    try:
        exchanges = conversation.incremental_exchanges(len(sessions))
        for key in sessions:
            session_exchanges = [
                exchange for exchange in exchanges if exchange.session == key
            ]
            for exchange in session_exchanges:
                speaker_results = []
                for speaker, speaker_id in speaker_ids.items():
                    result, latency = _timed(
                        lambda target=speaker, target_id=speaker_id: (
                            adapter.add_exchange(
                                _messages_for_speaker(exchange, target),
                                user_id=target_id,
                                speaker=target,
                                session=exchange.session,
                                session_date=exchange.session_date,
                                conversation_summary=conversation_summary,
                                recent_messages=list(recent_messages),
                            )
                        )
                    )
                    speaker_results.append(
                        {
                            "speaker": speaker,
                            "user_id": speaker_id,
                            "latency_ms": latency,
                            "result": result,
                        }
                    )
                ingestion.append(
                    {
                        "session": key,
                        "exchange_id": exchange.exchange_id,
                        "dialogue_ids": [
                            message.dialogue_id for message in exchange.messages
                        ],
                        "speaker_results": speaker_results,
                        "latency_ms": round(
                            sum(item["latency_ms"] for item in speaker_results), 3
                        ),
                    }
                )
                recent_messages.extend(
                    message.formatted() for message in exchange.messages
                )
                pending_summary_messages.extend(
                    message.formatted() for message in exchange.messages
                )
                pending_summary_exchange_ids.append(exchange.exchange_id)

                if (
                    len(pending_summary_exchange_ids)
                    == SUMMARY_PAIR_INTERVAL
                ):
                    summary_batch_index += 1
                    previous_summary = conversation_summary
                    conversation_summary, summary_ms = _timed(
                        lambda: summarizer.update(
                            previous_summary=previous_summary,
                            batch_index=summary_batch_index,
                            messages=list(pending_summary_messages),
                        )
                    )
                    summaries.append(
                        {
                            "batch_index": summary_batch_index,
                            "pair_count": len(
                                pending_summary_exchange_ids
                            ),
                            "exchange_ids": list(
                                pending_summary_exchange_ids
                            ),
                            "latency_ms": summary_ms,
                            "summary": conversation_summary,
                        }
                    )
                    pending_summary_messages.clear()
                    pending_summary_exchange_ids.clear()

        if pending_summary_exchange_ids:
            summary_batch_index += 1
            previous_summary = conversation_summary
            conversation_summary, summary_ms = _timed(
                lambda: summarizer.update(
                    previous_summary=previous_summary,
                    batch_index=summary_batch_index,
                    messages=list(pending_summary_messages),
                )
            )
            summaries.append(
                {
                    "batch_index": summary_batch_index,
                    "pair_count": len(pending_summary_exchange_ids),
                    "exchange_ids": list(pending_summary_exchange_ids),
                    "final_partial_batch": True,
                    "latency_ms": summary_ms,
                    "summary": conversation_summary,
                }
            )
        for question in questions:
            try:
                retrieval, retrieval_ms = _timed(
                    lambda item=question: _search_both_speakers(
                        adapter,
                        item.question,
                        speaker_ids=speaker_ids,
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
                        retrieval=_empty_retrieval(speaker_ids),
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
                judge_label = None
                judge_error = None
                judge_ms = None
                try:
                    judge_label, judge_ms = _timed(
                        lambda item=question, generated=answer: generator.judge(
                            question=item.question,
                            gold_answer=item.answer,
                            generated_answer=generated,
                        )
                    )
                except Exception as exc:
                    judge_error = f"{type(exc).__name__}: {exc}"
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
                        "llm_judge_status": (
                            "completed"
                            if judge_label is not None
                            else "failed"
                        ),
                        "llm_judge_label": judge_label,
                        "llm_judge_correct": (
                            judge_label == "CORRECT"
                            if judge_label is not None
                            else None
                        ),
                        "llm_judge_error": judge_error,
                        "llm_judge_latency_ms": judge_ms,
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
        for speaker_id in speaker_ids.values():
            try:
                adapter.delete_all(user_id=speaker_id)
            except Exception:
                pass
    return {
        "status": status,
        "error": error,
        "conversation_index": conversation.conversation_index,
        "sample_id": conversation.sample_id,
        "sessions": sessions,
        "speakers": speaker_ids,
        "ingestion": ingestion,
        "summary_updates": summaries,
        "final_conversation_summary": conversation_summary,
        "evaluations": evaluations,
    }


def run_benchmark(
    adapter: MemoryAdapter,
    generator: Generator,
    summarizer: Summarizer,
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
            summarizer,
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
    judged = [
        item
        for item in completed
        if item.get("llm_judge_label") in {"CORRECT", "WRONG"}
    ]
    judge_successes = sum(
        item["llm_judge_label"] == "CORRECT" for item in judged
    )
    judge_by_category: dict[str, list[bool]] = defaultdict(list)
    for item in judged:
        judge_by_category[item["category_name"]].append(
            item["llm_judge_label"] == "CORRECT"
        )
    control_values = [item["no_memory_official_f1"] for item in controlled]
    paired_memory = [item["official_f1"] for item in controlled]
    retrieval_latencies = [item["retrieval_latency_ms"] for item in completed]
    judge_latencies = [
        item["llm_judge_latency_ms"]
        for item in judged
        if item.get("llm_judge_latency_ms") is not None
    ]
    ingestion_latencies = [
        item["latency_ms"] for run in runs for item in run["ingestion"]
    ]

    def nonempty(value: Any) -> bool:
        if isinstance(value, dict):
            if "speakers" in value:
                return any(
                    nonempty(item.get("memories", []))
                    for item in value["speakers"]
                )
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
        "llm_judge_completed_count": len(judged),
        "llm_judge_failed_count": len(completed) - len(judged),
        "llm_judge_accuracy": (
            round(judge_successes / len(judged), 4) if judged else None
        ),
        "llm_judge_accuracy_95ci": wilson_interval(
            judge_successes, len(judged)
        ),
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
        "llm_judge_by_category": {
            label: {
                "count": len(values),
                "correct": sum(values),
                "accuracy": round(sum(values) / len(values), 4),
                "95ci": wilson_interval(sum(values), len(values)),
            }
            for label, values in sorted(judge_by_category.items())
        },
        "ingestion_latency_ms": {
            "p50": _p(ingestion_latencies, 0.50),
            "p95": _p(ingestion_latencies, 0.95),
        },
        "retrieval_latency_ms": {
            "p50": _p(retrieval_latencies, 0.50),
            "p95": _p(retrieval_latencies, 0.95),
        },
        "llm_judge_latency_ms": {
            "p50": _p(judge_latencies, 0.50),
            "p95": _p(judge_latencies, 0.95),
        },
    }
