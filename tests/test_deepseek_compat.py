from __future__ import annotations

from types import SimpleNamespace

from mem0_eval.integrations.deepseek import disable_deepseek_thinking


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


def test_deepseek_v4_internal_calls_disable_thinking() -> None:
    completions = FakeCompletions()
    llm = SimpleNamespace(
        config=SimpleNamespace(model="deepseek-v4-flash"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
    )
    assert disable_deepseek_thinking(llm) is True
    result = completions.create(model="deepseek-v4-flash")
    assert result["extra_body"] == {"thinking": {"type": "disabled"}}


def test_other_models_are_unchanged() -> None:
    completions = FakeCompletions()
    llm = SimpleNamespace(
        config=SimpleNamespace(model="gpt-4o-mini"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
    )
    assert disable_deepseek_thinking(llm) is False
