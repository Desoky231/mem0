from __future__ import annotations

import json

from mem0_eval.benchmarks.personamem.data import PersonaMemCase, sample_cases
from mem0_eval.benchmarks.personamem.protocol import (
    ingestion_batches,
    make_options,
    run_case,
    summarize_evaluations,
)
from mem0_eval.benchmarks.personamem.selector import parse_choice


def case(*, row: int = 1, updated: bool = False, kind: str = "neutral") -> PersonaMemCase:
    messages = (
        {"role": "user", "content": "What should I drink?"},
        {"role": "assistant", "content": "You might like mint tea."},
    )
    if updated:
        messages += (
            {"role": "user", "content": "Please forget that I like mint tea."},
            {"role": "assistant", "content": "Understood."},
        )
    return PersonaMemCase(
        row_index=row,
        persona_id=7,
        user_query="Suggest a warm drink.",
        correct_answer="Try hot chocolate.",
        incorrect_answers=("Try mint tea.", "Drink soda.", "Skip drinks."),
        preference="Do not remember 'Likes mint tea'" if updated else "Likes mint tea",
        previous_preference="Likes mint tea" if updated else None,
        messages=messages,
        preference_type=kind,
        updated=updated,
        who="self",
        sensitive=False,
        scenario="knowledge_query",
    )


class FakeAdapter:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def add(self, statement: str, *, user_id: str):
        self.statements.append(statement)
        return {"added": statement}

    def search(self, query: str, *, user_id: str):
        return {"results": [{"memory": "User asked to forget mint tea."}]}

    def get_all(self, *, user_id: str):
        return {"results": []}

    def delete_all(self, *, user_id: str):
        return None


class CorrectSelector:
    def select(self, *, query, options, retrieval):
        return {
            "choice": next(letter for letter, text in options if text == "Try hot chocolate."),
            "raw": "selected",
        }


class EmptySelector:
    def select(self, *, query, options, retrieval):
        return {"choice": None, "raw": ""}


def test_updated_case_is_ingested_in_event_order() -> None:
    assert len(ingestion_batches(case(updated=True))) == 2
    assert len(ingestion_batches(case(updated=False))) == 1


def test_options_are_deterministic() -> None:
    first = make_options(case(), 42)
    assert first == make_options(case(), 42)
    assert sorted(text for _, text in first[0]) == sorted(
        ["Try hot chocolate.", "Try mint tea.", "Drink soda.", "Skip drinks."]
    )


def test_round_robin_sampling_balances_preference_types() -> None:
    cases = [
        case(row=1, kind="a"),
        case(row=2, kind="a"),
        case(row=3, kind="b"),
        case(row=4, kind="b"),
    ]
    selected = sample_cases(cases, limit=2, seed=42)
    assert {item.preference_type for item in selected} == {"a", "b"}


def test_run_case_and_summary() -> None:
    evaluation = run_case(
        FakeAdapter(),
        case(updated=True),
        user_id="test-user",
        seed=42,
        selector=CorrectSelector(),
    )
    assert evaluation["mcq_correct"] is True
    assert evaluation["no_memory_mcq_correct"] is True
    assert evaluation["previous_preference_storage_leak"] is False
    summary = summarize_evaluations([evaluation])
    assert summary["mcq_accuracy"] == 1.0
    assert summary["all_requested_mcq_accuracy"] == 1.0
    assert summary["absolute_memory_accuracy_lift"] == 0.0
    assert summary["previous_preference_storage_leakage_rate"] == 0.0


def test_report_payload_is_json_serializable() -> None:
    evaluation = run_case(
        FakeAdapter(),
        case(),
        user_id="test-user",
        seed=42,
        selector=None,
    )
    json.dumps(evaluation)


def test_unparseable_selector_response_is_unscored() -> None:
    evaluation = run_case(
        FakeAdapter(),
        case(),
        user_id="test-user",
        seed=42,
        selector=EmptySelector(),
    )
    assert evaluation["mcq_correct"] is None


def test_choice_parser_rejects_reasoning_prose() -> None:
    assert parse_choice("A") == "A"
    assert parse_choice("Answer: C") == "C"
    assert parse_choice("We should choose A because it is best.") is None


def test_summary_ignores_unscorable_preference_recall() -> None:
    evaluation = run_case(
        FakeAdapter(), case(), user_id="test-user", seed=42, selector=None
    )
    evaluation["preference_token_recall"] = None
    summary = summarize_evaluations([evaluation])
    assert summary["mean_preference_token_recall"] is None
    assert summary["preference_token_recall_scorable_count"] == 0
