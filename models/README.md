# Models

This folder is the model-first control plane for `model-runner`.

Keep durable authored metadata here:
- backend config
- notes
- prompts
- templates
- compressor recipes

Do not treat this folder as the place for:
- weights
- quant outputs
- calibration datasets
- telemetry
- transcripts
- logs

Those belong outside the repo:
- `~/models` for weights and quantized artifacts
- `~/data/model-runner` for durable input datasets
- `~/runs/model-runner` for generated output

Each model gets its own directory, with backend-specific subfolders.

## Layout

- `models/<model_name>/<backend>/notes/`
- `models/<model_name>/<backend>/config/`
- `models/<model_name>/<backend>/templates/`
- `models/<model_name>/<backend>/prompts/`

Backends are typically:
- `hf` (Hugging Face / Transformers)
- `gguf` (llama-cpp-python)
- `ollama` (Ollama API)
- `exl2` (ExLlamaV2 / EXL2)

Shared assets:
- `models/_shared/<backend>/templates/`
- `models/_shared/<backend>/prompts/`

Templates:
- `models/_TEMPLATE/<backend>/...` contains starter files for new models.

## Mental Model

Use this split:

- repo `models/` = what you know about a model
- `~/models` = the actual model assets
- `~/data/model-runner` = reusable input data
- `~/runs/model-runner` = generated output

That keeps the repo navigable while preserving the model-first workflow.
