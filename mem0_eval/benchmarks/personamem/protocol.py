from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

from mem0_eval.benchmarks.eval_stats import (
    paired_difference_interval,
    wilson_interval,
)
from mem0_eval.benchmarks.memory_changes.metrics import percentile, rate

from .data import PersonaMemCase


class MemoryAdapter(Protocol):
    def add(self, statement: str, *, user_id: str) -> Any: ...

    def search(self, query: str, *, user_id: str) -> Any: ...

    def get_all(self, *, user_id: str) -> Any: ...

    def delete_all(self, *, user_id: str) -> Any: ...


class AnswerSelector(Protocol):
    def select(
        self,
        *,
        query: str,
        options: list[tuple[str, str]],
        retrieval: Any,
    ) -> dict[str, str | None]: ...


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, round((time.perf_counter() - started) * 1000, 3)


def _format_messages(messages: tuple[dict[str, str], ...]) -> str:
    return "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )


def ingestion_batches(case: PersonaMemCase) -> list[str]:
    """Preserve event order for deletion/update cases."""
    if case.updated and len(case.messages) >= 4:
        return [
            _format_messages(case.messages[:2]),
            _format_messages(case.messages[2:]),
        ]
    return [_format_messages(case.messages)]


def _normalized_tokens(text: str) -> set[str]:
    stop = {
        "about", "after", "again", "also", "and", "are", "been", "does", "from",
        "have", "into", "not", "that", "the", "their", "this", "with", "would",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2 and token not in stop
    }


def token_recall(expected: str, response: Any) -> float | None:
    expected_tokens = _normalized_tokens(expected)
    if not expected_tokens:
        return None
    observed = _normalized_tokens(json.dumps(response, ensure_ascii=False, default=str))
    return round(len(expected_tokens & observed) / len(expected_tokens), 4)


def contains_phrase(response: Any, phrase: str | None) -> bool | None:
    if not phrase:
        return None
    serialized = json.dumps(response, ensure_ascii=False, default=str).casefold()
    return phrase.casefold() in serialized


def make_options(case: PersonaMemCase, seed: int) -> tuple[list[tuple[str, str]], str]:
    choices = [case.correct_answer, *case.incorrect_answers]
    random.Random(f"{seed}:{case.row_index}").shuffle(choices)
    options = list(zip(("A", "B", "C", "D"), choices, strict=True))
    correct_letter = next(letter for letter, text in options if text == case.correct_answer)
    return options, correct_letter


def run_case(
    adapter: MemoryAdapter,
    case: PersonaMemCase,
    *,
    user_id: str,
    seed: int,
    selector: AnswerSelector | None,
    include_no_memory_control: bool = True,
) -> dict[str, Any]:
    ingestion: list[dict[str, Any]] = []
    try:
        for batch_index, statement in enumerate(ingestion_batches(case)):
            result, latency = _timed(lambda s=statement: adapter.add(s, user_id=user_id))
            ingestion.append(
                {
                    "batch_index": batch_index,
                    "latency_ms": latency,
                    "result": result,
                }
            )
        retrieval, retrieval_ms = _timed(
            lambda: adapter.search(case.user_query, user_id=user_id)
        )
        storage, storage_ms = _timed(lambda: adapter.get_all(user_id=user_id))
        options, correct_letter = make_options(case, seed)
        selection = (
            selector.select(query=case.user_query, options=options, retrieval=retrieval)
            if selector
            else {"choice": None, "raw": None}
        )
        control_selection = (
            selector.select(query=case.user_query, options=options, retrieval=[])
            if selector and include_no_memory_control
            else {"choice": None, "raw": None}
        )
        memory_correct = (
            selection["choice"] == correct_letter
            if selector is not None and selection["choice"] is not None
            else None
        )
        control_correct = (
            control_selection["choice"] == correct_letter
            if selector is not None and control_selection["choice"] is not None
            else None
        )
        return {
            "status": "completed",
            "case_id": case.case_id,
            "row_index": case.row_index,
            "persona_id": case.persona_id,
            "preference_type": case.preference_type,
            "updated": case.updated,
            "who": case.who,
            "sensitive": case.sensitive,
            "scenario": case.scenario,
            "user_query": case.user_query,
            "ingestion_batch_count": len(ingestion),
            "ingestion": ingestion,
            "retrieval_latency_ms": retrieval_ms,
            "storage_inspection_latency_ms": storage_ms,
            "retrieval": retrieval,
            "storage": storage,
            "preference_token_recall": token_recall(case.preference, retrieval),
            "previous_preference_retrieval_leak": contains_phrase(
                retrieval, case.previous_preference
            ),
            "previous_preference_storage_leak": contains_phrase(
                storage, case.previous_preference
            ),
            "options": [{"letter": letter, "text": text} for letter, text in options],
            "correct_letter": correct_letter,
            "selected_letter": selection["choice"],
            "selector_raw": selection["raw"],
            "mcq_correct": memory_correct,
            "no_memory_selected_letter": control_selection["choice"],
            "no_memory_selector_raw": control_selection["raw"],
            "no_memory_mcq_correct": control_correct,
            "memory_helped": (
                memory_correct and not control_correct
                if memory_correct is not None and control_correct is not None
                else None
            ),
            "memory_hurt": (
                control_correct and not memory_correct
                if memory_correct is not None and control_correct is not None
                else None
            ),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "case_id": case.case_id,
            "row_index": case.row_index,
            "error": f"{type(exc).__name__}: {exc}",
            "ingestion": ingestion,
        }
    finally:
        try:
            adapter.delete_all(user_id=user_id)
        except Exception:
            pass


def run_benchmark(
    adapter: MemoryAdapter,
    cases: list[PersonaMemCase],
    *,
    user_id_prefix: str,
    seed: int,
    selector: AnswerSelector | None,
    include_no_memory_control: bool = True,
) -> list[dict[str, Any]]:
    return [
        run_case(
            adapter,
            case,
            user_id=f"{user_id_prefix}_{case.row_index}",
            seed=seed,
            selector=selector,
            include_no_memory_control=include_no_memory_control,
        )
        for case in cases
    ]


def _p(values: list[float], probability: float) -> float | None:
    result = percentile(values, probability)
    return round(result, 3) if result is not None else None


def _group_accuracy(evaluations: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for item in evaluations:
        if item.get("mcq_correct") is not None:
            grouped[str(item[field])].append(bool(item["mcq_correct"]))
    return {
        label: {"count": len(values), "accuracy": rate(values)}
        for label, values in sorted(grouped.items())
    }


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in evaluations if item["status"] == "completed"]
    answered = [item for item in completed if item["mcq_correct"] is not None]
    controlled = [
        item for item in completed if item.get("no_memory_mcq_correct") is not None
    ]
    updated = [
        item
        for item in completed
        if item["updated"] and item["previous_preference_retrieval_leak"] is not None
    ]
    ingestion_latencies = [
        batch["latency_ms"] for item in completed for batch in item["ingestion"]
    ]
    retrieval_latencies = [item["retrieval_latency_ms"] for item in completed]
    preference_recalls = [
        item["preference_token_recall"]
        for item in completed
        if item["preference_token_recall"] is not None
    ]
    def nonempty(value: Any) -> bool:
        if isinstance(value, dict):
            value = value.get("results", [])
        return bool(value)

    all_requested_correct = sum(
        item.get("mcq_correct") is True for item in evaluations
    )
    return {
        "case_count": len(evaluations),
        "completed_count": len(completed),
        "failed_count": len(evaluations) - len(completed),
        "all_requested_mcq_accuracy": (
            all_requested_correct / len(evaluations) if evaluations else None
        ),
        "all_requested_mcq_accuracy_95ci": wilson_interval(
            all_requested_correct, len(evaluations)
        ),
        "mcq_answered_count": len(answered),
        "mcq_accuracy": rate(item["mcq_correct"] for item in answered),
        "mcq_accuracy_95ci": wilson_interval(
            sum(bool(item["mcq_correct"]) for item in answered), len(answered)
        ),
        "no_memory_control_count": len(controlled),
        "no_memory_mcq_accuracy": rate(
            item["no_memory_mcq_correct"] for item in controlled
        ),
        "no_memory_mcq_accuracy_95ci": wilson_interval(
            sum(bool(item["no_memory_mcq_correct"]) for item in controlled),
            len(controlled),
        ),
        "absolute_memory_accuracy_lift": (
            round(
                rate(item["mcq_correct"] for item in controlled)
                - rate(item["no_memory_mcq_correct"] for item in controlled),
                4,
            )
            if controlled
            else None
        ),
        "absolute_memory_accuracy_lift_95ci": paired_difference_interval(
            [float(item["mcq_correct"]) for item in controlled],
            [float(item["no_memory_mcq_correct"]) for item in controlled],
        ),
        "memory_help_rate": rate(item["memory_helped"] for item in controlled),
        "memory_hurt_rate": rate(item["memory_hurt"] for item in controlled),
        "updated_case_count": len(updated),
        "previous_preference_retrieval_leakage_rate": rate(
            item["previous_preference_retrieval_leak"] for item in updated
        ),
        "previous_preference_storage_leakage_rate": rate(
            item["previous_preference_storage_leak"] for item in updated
        ),
        "mean_preference_token_recall": (
            round(sum(preference_recalls) / len(preference_recalls), 4)
            if preference_recalls
            else None
        ),
        "preference_token_recall_scorable_count": len(preference_recalls),
        "nonempty_retrieval_rate": rate(
            nonempty(item["retrieval"]) for item in completed
        ),
        "nonempty_storage_rate": rate(
            nonempty(item["storage"]) for item in completed
        ),
        "ingestion_latency_ms": {
            "p50": _p(ingestion_latencies, 0.50),
            "p95": _p(ingestion_latencies, 0.95),
        },
        "retrieval_latency_ms": {
            "p50": _p(retrieval_latencies, 0.50),
            "p95": _p(retrieval_latencies, 0.95),
        },
        "accuracy_by_preference_type": _group_accuracy(completed, "preference_type"),
        "accuracy_by_updated": _group_accuracy(completed, "updated"),
        "accuracy_by_who": _group_accuracy(completed, "who"),
    }
