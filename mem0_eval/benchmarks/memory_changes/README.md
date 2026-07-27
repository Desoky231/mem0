# Memory-change checks

These checks ask one question: when a fact changes, does Mem0 stop returning
the old fact?

The 15 cases cover preferences, locations, schedules, roles, account settings,
and plans. Each case has a unique old identifier and replacement identifier so
stale retrieval can be scored exactly without an LLM judge.

Every case:

1. adds the original fact;
2. confirms that the original can be retrieved;
3. adds a conflicting replacement;
4. checks retrieval and storage for both identifiers;
5. deletes the isolated memory and checks that both identifiers are gone.

Run the checks with:

```bash
uv run python -m mem0_eval.run memory-changes
uv run --project mem0_eval/backends/graph \
  python -m mem0_eval.run memory-changes --backend graph
```

The definitions are in [`cases.json`](cases.json).
