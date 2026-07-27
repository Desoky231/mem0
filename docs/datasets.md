# Datasets

## LoCoMo

LoCoMo contains ten long conversations split across dated sessions. Its
questions test:

- direct fact recall;
- temporal reasoning;
- combining facts from several sessions;
- broad questions requiring several supporting details;
- resistance to false assumptions.

This project evaluates categories 1–4 and only selects questions whose cited
conversation turns were actually stored.

Local file: `data/locomo10.json`

Source: <https://github.com/snap-research/locomo>

## PersonaMem-v2

PersonaMem-v2 tests whether previous conversations help an assistant answer in
a way that matches a person's preferences. Each evaluated row provides a later
question, one correct answer, three alternatives, and a related conversation.

The experiment stores only the conversation. It does not store the correct
answer, preference label, persona summary, or other answer-revealing fields.

Local file: `data/personamem_v2.json`

Source: <https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2>

Download both datasets with:

```bash
uv run python -m mem0_eval.run download
```
