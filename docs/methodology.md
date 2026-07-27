# Methodology and limitations

## What is compared

Text memory uses `mem0ai==2.0.14`. Graph memory uses the historical
`mem0ai==0.1.45` `MemoryGraph` implementation. The graph API used by the
original implementation is not available in the current package, so the two
baselines cannot share one dependency environment.

This means the evaluation compares complete implementations, not only vector
storage versus graph storage.

## Controls

- LoCoMo and PersonaMem use deterministic samples with seed 42.
- Both backends receive the same sampled inputs and questions.
- Answer generation uses the same model.
- A paired no-memory answer is recorded to estimate whether retrieved memory
  improves the result.
- Every dataset row or conversation uses an isolated memory scope.

## LoCoMo ingestion

LoCoMo sessions are processed in chronological order, but a complete session is
not submitted as one memory request. Each session is divided into
non-overlapping dialogue pairs. For every pair:

1. The two messages are submitted as the new exchange.
2. Each speaker receives a separate memory update from that speaker's
   perspective.
3. `mem0ai==2.0.14` supplies its native previous-10-message history for each
   stable speaker scope, including messages from an earlier LoCoMo session.
4. A cumulative summary of the whole conversation so far is supplied as global
   context.

After every five pairs, DeepSeek combines the previous summary with those five
pairs. The result is an 800-token-maximum key-knowledge summary organized by
person and shared timeline, not a transcript summary. A final partial batch is
summarized after ingestion. The refresh is synchronous in this offline
benchmark. The paper describes an asynchronous periodic refresher but does not
publish its cadence or summarization prompt, so five pairs is a documented
benchmark choice.

Questions search both speaker scopes. The answer prompt receives the two
timestamped result groups separately, prioritizes newer conflicting memories,
resolves relative dates, and requests an answer shorter than 5–6 words. This is
the paper's appendix results-generation prompt. After official token F1 is
calculated, the same generated answer is evaluated with the paper's binary
LLM-judge prompt. Judge failures are recorded without discarding the F1 result.

## Memory changes

The memory-change experiment uses unique exact markers. It reports old facts
found in search separately from old facts still present in storage. Explicit
scope deletion is an engineering check, not a claim about the paper's
conversational conflict-resolution behavior.

## Interpretation

The checked-in report describes results from the earlier session-level LoCoMo
runner. It should not be presented as evidence for the new incremental
protocol until that protocol is run and the report is regenerated. Even after
regeneration, this is paper-aligned rather than an exact reproduction: it uses
DeepSeek, different embeddings, newer text-memory internals, and a historical
graph package.
Historical graph extraction can vary between runs even at temperature zero, so
larger repeated runs are needed for stable estimates.
