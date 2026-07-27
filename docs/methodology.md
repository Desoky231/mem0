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

## Memory changes

The memory-change experiment uses unique exact markers. It reports old facts
found in search separately from old facts still present in storage. Explicit
scope deletion is an engineering check, not a claim about the paper's
conversational conflict-resolution behavior.

## Interpretation

The checked-in report describes small pilot samples. It should not be presented
as a full reproduction of published Mem0, LoCoMo, or PersonaMem-v2 scores.
Historical graph extraction can vary between runs even at temperature zero, so
larger repeated runs are needed for stable estimates.
