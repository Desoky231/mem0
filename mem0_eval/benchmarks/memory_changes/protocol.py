from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from .metrics import contains_marker, percentile, rate
from .case import MemoryChangeCase


class MemoryAdapter(Protocol):
    def add(self, statement: str, *, user_id: str) -> Any: ...

    def search(self, query: str, *, user_id: str) -> Any: ...

    def get_all(self, *, user_id: str) -> Any: ...

    def delete_all(self, *, user_id: str) -> Any: ...


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1000


def _result_items(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        response = response.get("results", [])
    if not isinstance(response, list):
        return []
    return [item for item in response if isinstance(item, dict)]


def _stage(
    *,
    name: str,
    operation: str,
    operation_result: Any,
    operation_ms: float,
    retrieval_result: Any,
    retrieval_ms: float,
    storage_result: Any,
    storage_ms: float,
    current_markers: Iterable[str],
    stale_markers: Iterable[str],
) -> dict[str, Any]:
    return {
        "stage": name,
        "operation": operation,
        "operation_latency_ms": round(operation_ms, 3),
        "retrieval_latency_ms": round(retrieval_ms, 3),
        "storage_inspection_latency_ms": round(storage_ms, 3),
        "current_marker_hit": any(
            contains_marker(retrieval_result, marker) for marker in current_markers
        ),
        "current_marker_stored": any(
            contains_marker(storage_result, marker) for marker in current_markers
        ),
        "stale_memory_leakage": any(
            contains_marker(retrieval_result, marker) for marker in stale_markers
        ),
        "stale_storage_leakage": any(
            contains_marker(storage_result, marker) for marker in stale_markers
        ),
        "operation_result": operation_result,
        "retrieval_result": retrieval_result,
        "storage_result": storage_result,
    }


def run_case(
    adapter: MemoryAdapter,
    case: MemoryChangeCase,
    *,
    user_id: str,
) -> dict[str, Any]:
    """Run ADD -> inferred UPDATE -> explicit DELETE for one isolated user scope."""
    add_result, add_ms = _timed(
        lambda: adapter.add(case.baseline_statement, user_id=user_id)
    )
    baseline_retrieval, baseline_search_ms = _timed(
        lambda: adapter.search(case.query, user_id=user_id)
    )
    baseline_storage, baseline_storage_ms = _timed(
        lambda: adapter.get_all(user_id=user_id)
    )
    stages = [
        _stage(
            name="baseline",
            operation="inferred_add",
            operation_result=add_result,
            operation_ms=add_ms,
            retrieval_result=baseline_retrieval,
            retrieval_ms=baseline_search_ms,
            storage_result=baseline_storage,
            storage_ms=baseline_storage_ms,
            current_markers=[case.old_marker],
            stale_markers=[],
        )
    ]

    update_result, update_ms = _timed(
        lambda: adapter.add(case.updated_statement, user_id=user_id)
    )
    update_retrieval, update_search_ms = _timed(
        lambda: adapter.search(case.query, user_id=user_id)
    )
    update_storage, update_storage_ms = _timed(
        lambda: adapter.get_all(user_id=user_id)
    )
    stages.append(
        _stage(
            name="update",
            operation="inferred_update_via_add",
            operation_result=update_result,
            operation_ms=update_ms,
            retrieval_result=update_retrieval,
            retrieval_ms=update_search_ms,
            storage_result=update_storage,
            storage_ms=update_storage_ms,
            current_markers=[case.new_marker],
            stale_markers=[case.old_marker],
        )
    )

    delete_result, delete_ms = _timed(
        lambda: adapter.delete_all(user_id=user_id)
    )
    delete_retrieval, delete_search_ms = _timed(
        lambda: adapter.search(case.query, user_id=user_id)
    )
    delete_storage, delete_storage_ms = _timed(
        lambda: adapter.get_all(user_id=user_id)
    )
    stages.append(
        _stage(
            name="delete",
            operation="explicit_delete_all_isolated_scope",
            operation_result={
                "listed_before_delete": update_storage,
                "delete_result": delete_result,
            },
            operation_ms=delete_ms,
            retrieval_result=delete_retrieval,
            retrieval_ms=delete_search_ms,
            storage_result=delete_storage,
            storage_ms=delete_storage_ms,
            current_markers=[],
            stale_markers=[case.old_marker, case.new_marker],
        )
    )

    return {
        "case_id": case.case_id,
        "subject": case.subject,
        "user_id": user_id,
        "protocol": "inferred-add__inferred-update-via-add__explicit-delete",
        "stages": stages,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_stages = [
        stage
        for run in runs
        for stage in run["stages"]
        if stage["stage"] == "baseline"
    ]
    update_stages = [
        stage
        for run in runs
        for stage in run["stages"]
        if stage["stage"] == "update"
    ]
    delete_stages = [
        stage
        for run in runs
        for stage in run["stages"]
        if stage["stage"] == "delete"
    ]
    mutation_stages = update_stages + delete_stages
    retrieval_latencies = [
        stage["retrieval_latency_ms"] for run in runs for stage in run["stages"]
    ]
    operation_latencies = [
        stage["operation_latency_ms"] for run in runs for stage in run["stages"]
    ]
    return {
        "case_count": len(runs),
        "update_stale_leakage_rate": rate(
            stage["stale_memory_leakage"] for stage in update_stages
        ),
        "update_stale_storage_leakage_rate": rate(
            stage["stale_storage_leakage"] for stage in update_stages
        ),
        "delete_stale_leakage_rate": rate(
            stage["stale_memory_leakage"] for stage in delete_stages
        ),
        "delete_stale_storage_leakage_rate": rate(
            stage["stale_storage_leakage"] for stage in delete_stages
        ),
        "overall_mutation_stale_leakage_rate": rate(
            stage["stale_memory_leakage"] for stage in mutation_stages
        ),
        "overall_mutation_stale_storage_leakage_rate": rate(
            stage["stale_storage_leakage"] for stage in mutation_stages
        ),
        "baseline_current_marker_hit_rate": rate(
            stage["current_marker_hit"] for stage in baseline_stages
        ),
        "baseline_current_marker_stored_rate": rate(
            stage["current_marker_stored"] for stage in baseline_stages
        ),
        "update_current_marker_hit_rate": rate(
            stage["current_marker_hit"] for stage in update_stages
        ),
        "retrieval_latency_ms": {
            "p50": _rounded_percentile(retrieval_latencies, 0.50),
            "p95": _rounded_percentile(retrieval_latencies, 0.95),
        },
        "operation_latency_ms": {
            "p50": _rounded_percentile(operation_latencies, 0.50),
            "p95": _rounded_percentile(operation_latencies, 0.95),
        },
    }


def _rounded_percentile(values: list[float], probability: float) -> float | None:
    value = percentile(values, probability)
    return round(value, 3) if value is not None else None
