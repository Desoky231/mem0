from __future__ import annotations

from typing import Any

import pytest

from mem0_eval.benchmarks.memory_changes.case import MemoryChangeCase
from mem0_eval.benchmarks.memory_changes.metrics import flatten_text, percentile
from mem0_eval.benchmarks.memory_changes.protocol import run_case, summarize_runs


CASE = MemoryChangeCase(
    case_id="drink",
    subject="Avery's drink",
    baseline_statement="Avery drinks tea under ORCHID71.",
    updated_statement="Avery drinks coffee under COBALT92.",
    query="What does Avery drink?",
    old_marker="ORCHID71",
    new_marker="COBALT92",
)


class FakeMemory:
    def __init__(self, *, resolve_updates: bool) -> None:
        self.resolve_updates = resolve_updates
        self.items: dict[str, str] = {}
        self.calls = 0

    def add(self, statement: str, *, user_id: str) -> dict[str, Any]:
        self.calls += 1
        if self.resolve_updates and self.items:
            memory_id = next(iter(self.items))
            self.items[memory_id] = statement
            event = "UPDATE"
        else:
            memory_id = str(self.calls)
            self.items[memory_id] = statement
            event = "ADD"
        return {"results": [{"id": memory_id, "memory": statement, "event": event}]}

    def search(self, query: str, *, user_id: str) -> dict[str, Any]:
        return {
            "results": [
                {"id": memory_id, "memory": text}
                for memory_id, text in self.items.items()
            ]
        }

    def get_all(self, *, user_id: str) -> dict[str, Any]:
        return self.search("", user_id=user_id)

    def delete_all(self, *, user_id: str) -> dict[str, str]:
        self.items.clear()
        return {"message": "deleted"}


def test_protocol_detects_update_leak_and_clean_delete() -> None:
    result = run_case(FakeMemory(resolve_updates=False), CASE, user_id="isolated")
    stages = {stage["stage"]: stage for stage in result["stages"]}
    assert stages["baseline"]["current_marker_hit"] is True
    assert stages["update"]["current_marker_hit"] is True
    assert stages["update"]["stale_memory_leakage"] is True
    assert stages["update"]["stale_storage_leakage"] is True
    assert stages["delete"]["stale_memory_leakage"] is False
    assert stages["delete"]["stale_storage_leakage"] is False


def test_protocol_accepts_resolved_update() -> None:
    result = run_case(FakeMemory(resolve_updates=True), CASE, user_id="isolated")
    summary = summarize_runs([result])
    assert summary["update_stale_leakage_rate"] == 0.0
    assert summary["update_stale_storage_leakage_rate"] == 0.0
    assert summary["delete_stale_leakage_rate"] == 0.0
    assert summary["baseline_current_marker_hit_rate"] == 1.0
    assert summary["update_current_marker_hit_rate"] == 1.0


def test_case_rejects_old_marker_in_update_statement() -> None:
    with pytest.raises(ValueError, match="must not repeat old_marker"):
        MemoryChangeCase(
            case_id="bad",
            subject="subject",
            baseline_statement="Old ORCHID71",
            updated_statement="Not ORCHID71, now COBALT92",
            query="current?",
            old_marker="ORCHID71",
            new_marker="COBALT92",
        ).validate()


def test_nested_graph_like_response_is_scored() -> None:
    response = {
        "results": [],
        "relations": [{"source": "Avery", "relationship": "uses", "target": "ORCHID71"}],
    }
    assert "ORCHID71" in flatten_text(response)


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0], 0.5) == 15.0
    assert percentile([], 0.95) is None
