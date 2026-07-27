# Methodology and limitations

## What is compared

Text memory uses `mem0ai==2.0.14`. Mem0g uses that same text implementation
plus a paper-specific graph layer implemented directly over Neo4j. The legacy
`mem0ai==0.1.45` `MemoryGraph` package is not part of current runs.

This keeps the text component and dependency environment shared, making the
graph comparison an augmentation of the working text backend.

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

For Mem0g, text extraction and two-stage entity/relation extraction run
concurrently for every pair. Context is available to the graph extractor only
for resolving references; it cannot be the sole evidence for a relation.
Graph nodes and relations include dialogue/message provenance, observation and
session metadata, creation and validity timestamps, and status. Contradicted
relations are marked obsolete with a `valid_to` value instead of being deleted.

Graph retrieval merges entity-centric neighborhood scoring with semantic
triplet scoring. Questions search both speaker scopes. The Mem0g answer prompt
receives text memories and graph relations separately for both speakers. It
prioritizes newer conflicting memories, resolves relative dates, and requests
an answer shorter than 5–6 words. After official token F1 is calculated, the
same generated answer is evaluated with the paper's binary LLM-judge prompt.
Judge failures are recorded without discarding the F1 result.

## Memory changes

The memory-change experiment uses unique exact markers. It reports old facts
found in search separately from old facts still present in storage. Explicit
scope deletion is an engineering check, not a claim about the paper's
conversational conflict-resolution behavior.

## Interpretation

The checked-in graph report describes the retired historical graph backend. It
should not be presented as evidence for the composite Mem0g implementation
until a new graph run is completed and the report is regenerated. Even then,
this is paper-aligned rather than an exact reproduction: it uses DeepSeek,
different embeddings, a documented five-pair summary cadence, and
repository-defined similarity thresholds because the paper does not publish
all prompts, thresholds, or refresh timing. Graph extraction can vary between
runs even at temperature zero, so larger repeated runs are needed for stable
estimates.
