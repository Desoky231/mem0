# PersonaMem-v2 experiment

PersonaMem-v2 asks whether an assistant can infer a person's preference from an
earlier conversation and use it in a later answer.

For each sampled row, this project:

1. stores only the related conversation;
2. searches memory using the later question;
3. asks the answer model to choose from the official answer options;
4. repeats the question with no retrieved memory as a control;
5. deletes the isolated memory.

Private answer fields and persona summaries are never stored as memory.
Sensitive and multimodal rows are excluded by default.

## Run

Preview the selected rows without API calls:

```bash
uv run python -m mem0_eval.run personamem --dry-run
uv run --project mem0_eval/backends/graph \
  python -m mem0_eval.run personamem --backend graph --dry-run
```

Run the small default sample:

```bash
uv run python -m mem0_eval.run personamem
uv run --project mem0_eval/backends/graph \
  python -m mem0_eval.run personamem --backend graph
```

## Current paired result

The current paired run used the same 60 balanced rows for each backend.

| Metric | Text v2.0.14 | Graph v0.1.45 |
|---|---:|---:|
| Accuracy with memory | 48.3% | 38.3% |
| Accuracy without memory | 21.7% | 21.7% |
| Memory improvement | +26.7 points | +16.7 points |
| Non-empty retrieval | 76.7% | 66.7% |

This is a small diagnostic sample, not a full PersonaMem-v2 reproduction. See
the [full report](reports/mem0_paired_baseline_evaluation.html) for confidence
intervals and limitations.
