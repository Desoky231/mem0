# Mem0 memory evaluation

This project checks how well Mem0 remembers conversations and what happens when
a remembered fact changes.

It contains three experiments:

| Experiment | Question it answers |
|---|---|
| LoCoMo | Can the memory answer questions about long, multi-session conversations? |
| PersonaMem-v2 | Does remembered personal context improve the assistant's answer? |
| Memory changes | Does an old fact remain after it is updated or deleted? |

The project compares two Mem0 implementations:

- **Text memory:** the current `mem0ai==2.0.14` package with local vector search.
- **Graph memory:** a Mem0g composite backend that combines the same text
  memory with a temporal Neo4j graph in
  [`mem0_eval/backends/graph/`](mem0_eval/backends/graph/).

The graph behavior is implemented directly in this repository rather than
through the obsolete `mem0ai==0.1.45` graph API. Both backends now use the same
current Mem0 text implementation; graph runs add entity/relation extraction,
temporal Neo4j storage, and dual retrieval.

## Setup

Requirements: Python 3.11, [`uv`](https://docs.astral.sh/uv/), and a DeepSeek API
key.

```bash
uv sync
cp .env.example .env
```

Add your key to `.env`, then download the datasets:

```bash
uv run python -m mem0_eval.run download
```

## Run the text-memory experiments

Each command has useful small-sample defaults, so parameters are optional.

```bash
uv run python -m mem0_eval.run locomo
uv run python -m mem0_eval.run personamem
uv run python -m mem0_eval.run memory-changes
```

Use `--help` only when you want to change the sample size or another setting:

```bash
uv run python -m mem0_eval.run locomo --help
```

## Run the graph-memory experiments

Graph memory additionally requires Neo4j:

```bash
docker compose up -d
uv run python -m mem0_eval.run check-graph
uv run python -m mem0_eval.run locomo --backend graph
uv run python -m mem0_eval.run personamem --backend graph
uv run python -m mem0_eval.run memory-changes --backend graph
```

Clear all local Neo4j benchmark data without stopping the container:

```bash
uv run python -m mem0_eval.backends.graph.clear_neo4j
```

See [`mem0_eval/backends/graph/README.md`](mem0_eval/backends/graph/README.md)
for the graph pipeline and metadata.

## Project layout

```text
.
├── mem0_eval/
│   ├── backends/
│   │   ├── text/
│   │   └── graph/
│   ├── benchmarks/
│   │   ├── locomo/
│   │   ├── personamem/
│   │   ├── memory_changes/
│   │   └── eval_stats.py
│   ├── integrations/
│   │   └── deepseek.py
│   └── run.py             # One command interface for every combination
├── tests/                 # Network-free automated tests
├── docs/                  # Optional background and generated reports
└── results/               # Generated locally; not committed
```

Most users only need `python -m mem0_eval.run`.

## Memory-change case catalog

The 15 included cases cover preferences, locations, schedules, roles, account
settings, and future plans.

Every case adds an original fact, adds a conflicting replacement, then deletes
the isolated memory. Exact identifiers make stale facts easy to detect without
an LLM judge. The cases and complete protocol are in
[`mem0_eval/benchmarks/memory_changes/`](mem0_eval/benchmarks/memory_changes/README.md).

## Results and limitations

The current results are small pilot experiments, not full benchmark
reproductions. The main generated report is available in
[`docs/reports/mem0_paired_baseline_evaluation.html`](docs/reports/mem0_paired_baseline_evaluation.html).

LoCoMo is ingested chronologically as message pairs. Each speaker has a
separate memory scope. A rolling window of the previous 10 messages continues
across session boundaries using Mem0's native history. Every five pairs,
DeepSeek updates an 800-token key-knowledge summary organized by person and
shared timeline. Answers use the paper's results-generation prompt and are
scored with both token F1 and its LLM-judge prompt.

Run the network-free checks with:

```bash
uv run pytest -q
```
