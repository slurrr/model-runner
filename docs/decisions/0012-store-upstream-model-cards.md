# 0012 - Store upstream model cards as `notes/model_card.md`

Date: 2026-03-07

## Context
Upstream model cards (typically `README.md` in the model folder / HF repo) are useful references for:
- architecture + context length claims
- recommended runtimes (Transformers vs vLLM vs SGLang)
- modality support (vision/audio)
- special prompting/tooling formats

But keeping them as `README.md` inside this repo conflicts with our convention that `notes/README.md` is the “repo-local” notes file for how to run the model here.

## Decision
When we copy/import an upstream model card into this repo, store it as:
- `models/<model>/<backend>/notes/model_card.md`

Keep our repo-local notes at:
- `models/<model>/<backend>/notes/README.md`

Automation:
- `scripts/model add` may copy a local model folder’s `README.md` into `notes/model_card.md` for HF models.

## Consequences
- Notes remain consistent and easy to find.
- Upstream model documentation is preserved without clobbering local notes.
- Onboarding new HF models becomes more repeatable.

