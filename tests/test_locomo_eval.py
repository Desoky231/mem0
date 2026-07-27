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
from mem0_eval.benchmarks.locomo.generator import (
    GRAPH_RESULTS_GENERATION_PROMPT,
    RESULTS_GENERATION_PROMPT,
    _judge_label,
)
from mem0_eval.benchmarks.locomo.protocol import summarize_evaluations
from mem0_eval.benchmarks.locomo.protocol import run_conversation
from mem0_eval.backends.text.adapter import TextMemoryAdapter


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


def test_incremental_exchanges_preserve_session_boundaries_and_dates() -> None:
    conversation = LoCoMoConversation(
        conversation_index=0,
        sample_id="x",
        raw_conversation={
            "speaker_a": "Ada",
            "speaker_b": "Ben",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {"speaker": "Ada", "dia_id": "D1:1", "text": "First"},
                {"speaker": "Ben", "dia_id": "D1:2", "text": "Second"},
                {"speaker": "Ada", "dia_id": "D1:3", "text": "Unpaired"},
            ],
            "session_2_date_time": "2 January 2024",
            "session_2": [
                {"speaker": "Ben", "dia_id": "D2:1", "text": "New session"},
            ],
        },
        questions=(),
    )

    exchanges = conversation.incremental_exchanges(2)

    assert [item.exchange_id for item in exchanges] == [
        "session_1-exchange-0",
        "session_1-exchange-1",
        "session_2-exchange-0",
    ]
    assert [len(item.messages) for item in exchanges] == [2, 1, 1]
    assert exchanges[-1].messages[0].formatted().startswith(
        "[2 January 2024; D2:1]"
    )


def test_incremental_runner_carries_context_across_sessions() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.additions = []
            self.deleted = []

        def add_exchange(self, messages, **context):
            self.additions.append((messages, context))
            return {"results": []}

        def search(self, query, *, user_id):
            return [{"memory": f"{user_id}:{query}"}]

        def delete_all(self, *, user_id):
            self.deleted.append(user_id)

    class FakeGenerator:
        def answer(self, *, question, category, retrieval):
            return "answer"

        def judge(self, *, question, gold_answer, generated_answer):
            return "CORRECT"

    class FakeSummarizer:
        def __init__(self) -> None:
            self.calls = []

        def update(self, **context):
            self.calls.append(context)
            return f"summary through batch {context['batch_index']}"

    conversation = LoCoMoConversation(
        conversation_index=0,
        sample_id="x",
        raw_conversation={
            "speaker_a": "Ada",
            "speaker_b": "Ben",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {
                    "speaker": "Ada" if index % 2 == 0 else "Ben",
                    "dia_id": f"D1:{index + 1}",
                    "text": f"Session one message {index + 1}",
                }
                for index in range(6)
            ],
            "session_2_date_time": "2 January 2024",
            "session_2": [
                {
                    "speaker": "Ada" if index % 2 == 0 else "Ben",
                    "dia_id": f"D2:{index + 1}",
                    "text": f"Session two message {index + 1}",
                }
                for index in range(6)
            ],
        },
        questions=(),
    )
    adapter = FakeAdapter()
    summarizer = FakeSummarizer()

    result = run_conversation(
        adapter,
        FakeGenerator(),
        summarizer,
        conversation,
        user_id="conversation",
        session_count=2,
        questions_per_category=0,
        categories=(1, 2, 3, 4),
        seed=42,
        include_no_memory_control=False,
    )

    assert result["status"] == "completed"
    assert len(adapter.additions) == 12
    first_ada_messages, first_ada_context = adapter.additions[0]
    assert [message["role"] for message in first_ada_messages] == [
        "user",
        "assistant",
    ]
    sixth_pair_context = adapter.additions[10][1]
    assert sixth_pair_context["conversation_summary"] == (
        "summary through batch 1"
    )
    assert len(sixth_pair_context["recent_messages"]) == 10
    assert [call["batch_index"] for call in summarizer.calls] == [1, 2]
    assert [len(call["messages"]) for call in summarizer.calls] == [10, 2]
    assert [item["pair_count"] for item in result["summary_updates"]] == [5, 1]
    assert result["summary_updates"][-1]["final_partial_batch"] is True
    assert sorted(adapter.deleted) == [
        "conversation_speaker_1",
        "conversation_speaker_2",
    ]


def test_paper_prompts_and_judge_json_parser() -> None:
    assert "Memories for user {speaker_1_user_id}" in RESULTS_GENERATION_PROMPT
    assert "The answer should be less than 5-6 words." in RESULTS_GENERATION_PROMPT
    assert (
        "Relations for user {speaker_1_user_id}"
        in GRAPH_RESULTS_GENERATION_PROMPT
    )
    assert (
        "{speaker_1_graph_memories}"
        in GRAPH_RESULTS_GENERATION_PROMPT
    )
    assert (
        "Relations for user {speaker_2_user_id}"
        in GRAPH_RESULTS_GENERATION_PROMPT
    )
    assert (
        "{speaker_2_graph_memories}"
        in GRAPH_RESULTS_GENERATION_PROMPT
    )
    assert _judge_label('{"label": "correct"}') == "CORRECT"


def test_text_adapter_relies_on_mem0_native_recent_message_history() -> None:
    class FakeMemory:
        def __init__(self) -> None:
            self.arguments = None

        def add(self, messages, **arguments):
            self.arguments = arguments
            return {"results": []}

    memory = FakeMemory()
    adapter = TextMemoryAdapter(memory, top_k=10, threshold=0.0)
    adapter.add_exchange(
        [{"role": "user", "content": "Ada: hello"}],
        user_id="ada",
        speaker="Ada",
        session="session_2",
        session_date="2 January 2024",
        conversation_summary="Ada moved to Cairo.",
        recent_messages=["This must not be duplicated in the prompt."],
    )

    prompt = memory.arguments["prompt"]
    assert "Key-knowledge summary:\nAda moved to Cairo." in prompt
    assert "This must not be duplicated in the prompt." not in prompt
    assert "native previous-10-message window" in prompt


def test_runner_records_f1_then_llm_judge_result() -> None:
    class FakeAdapter:
        def add_exchange(self, messages, **context):
            return {"results": []}

        def search(self, query, *, user_id):
            return [{"memory": "Ada moved to Cairo"}]

        def delete_all(self, *, user_id):
            return None

    class FakeGenerator:
        def answer(self, *, question, category, retrieval):
            return "Cairo"

        def judge(self, *, question, gold_answer, generated_answer):
            assert generated_answer == "Cairo"
            return "CORRECT"

    class FakeSummarizer:
        def update(self, **context):
            return "Ada moved to Cairo."

    conversation = LoCoMoConversation(
        conversation_index=0,
        sample_id="x",
        raw_conversation={
            "speaker_a": "Ada",
            "speaker_b": "Ben",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {"speaker": "Ada", "dia_id": "D1:1", "text": "I moved to Cairo"},
                {"speaker": "Ben", "dia_id": "D1:2", "text": "That is great"},
            ],
        },
        questions=(
            LoCoMoQuestion(
                0,
                0,
                "Where did Ada move?",
                "Cairo",
                ("D1:1",),
                1,
            ),
        ),
    )

    result = run_conversation(
        FakeAdapter(),
        FakeGenerator(),
        FakeSummarizer(),
        conversation,
        user_id="conversation",
        session_count=1,
        questions_per_category=1,
        categories=(1,),
        seed=42,
        include_no_memory_control=False,
    )

    evaluation = result["evaluations"][0]
    assert evaluation["official_f1"] == 1.0
    assert evaluation["llm_judge_label"] == "CORRECT"
    assert evaluation["llm_judge_correct"] is True


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
        "llm_judge_label": "WRONG",
        "llm_judge_latency_ms": 3.0,
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
    assert summary["llm_judge_accuracy"] == 0.0
    assert summary["llm_judge_latency_ms"]["p50"] == 3.0
