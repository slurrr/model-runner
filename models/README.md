# Models

This folder is the model-first workspace for metadata and tooling around each model you test.

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
