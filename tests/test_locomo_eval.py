from __future__ import annotations

from mem0_eval.benchmarks.eval_stats import (
    paired_difference_interval,
    wilson_interval,
)
from mem0_eval.benchmarks.locomo.data import (
    LoCoMoConversation,
    LoCoMoQuestion,
    sample_questions,
)
from mem0_eval.benchmarks.locomo.metrics import (
    answer_token_recall,
    official_locomo_f1,
    token_f1,
)
from mem0_eval.benchmarks.locomo.protocol import summarize_evaluations


def test_official_f1_normalizes_and_stems() -> None:
    assert token_f1("The adopted cats", "adopting cat") == 1.0
    assert official_locomo_f1("Psychology", "Psychology; because x", 3) == 1.0


def test_category_one_multi_answer_scoring() -> None:
    score = official_locomo_f1("hiking, painting", "painting, hiking", 1)
    assert score == 1.0


def test_retrieval_answer_token_recall() -> None:
    assert answer_token_recall("Caroline researched adoption agencies", "Adoption agencies") == 1.0


def test_question_sampling_requires_ingested_evidence() -> None:
    conversation = LoCoMoConversation(
        conversation_index=0,
        sample_id="x",
        raw_conversation={
            "session_1": [],
            "session_1_date_time": "date",
            "session_2": [],
            "session_2_date_time": "date",
        },
        questions=(
            LoCoMoQuestion(0, 0, "early", "a", ("D1:1",), 1),
            LoCoMoQuestion(0, 1, "late", "b", ("D3:1",), 1),
        ),
    )
    sampled = sample_questions(
        conversation,
        session_count=2,
        questions_per_category=5,
        categories=(1,),
        seed=42,
    )
    assert [item.question for item in sampled] == ["early"]


def test_confidence_interval_helpers() -> None:
    assert wilson_interval(5, 10) == [0.2366, 0.7634]
    assert paired_difference_interval([1, 1], [0, 0]) == [1.0, 1.0]


def test_locomo_summary_reports_coverage_and_abstention() -> None:
    evaluation = {
        "status": "completed",
        "category_name": "single_hop",
        "official_f1": 0.0,
        "no_memory_official_f1": 0.0,
        "retrieval_answer_token_recall": 0.0,
        "retrieval": [{"source": "a", "relationship": "r", "destination": "b"}],
        "generated_answer": "Not mentioned in the retrieved memory.",
        "retrieval_latency_ms": 1.0,
    }
    summary = summarize_evaluations(
        [
            {
                "status": "completed",
                "evaluations": [evaluation],
                "ingestion": [{"latency_ms": 2.0}],
            }
        ]
    )
    assert summary["nonempty_retrieval_rate"] == 1.0
    assert summary["answer_abstention_rate"] == 1.0
