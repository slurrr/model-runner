# EXL2 / ExLlamaV2 setup notes (WSL + CUDA)

This repo can run EXL2 models via ExLlamaV2 (TUI backend: `--backend exl2`), but ExLlamaV2 requires a compiled CUDA extension.
This doc is a pragmatic checklist for getting a working build under WSL.

## Quick checks

In the same venv you’ll run `tui` from:

```bash
. .venv/bin/activate
python -c "import torch; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available())"
nvcc --version
```

Rule of thumb:
- `torch.version.cuda` **major.minor** should match `nvcc` major.minor (or be very close).

Example mismatch:
- torch says `12.8` but nvcc is `12.4` → extension builds may fail or produce runtime link errors.

## Required build dependency

ExLlamaV2 uses PyTorch’s C++ extension build tooling which requires Ninja:

```bash
pip install ninja
```

If you see:
- `RuntimeError: Ninja is required to load C++ extensions`
…that’s the fix.

## Installing ExLlamaV2

If you have a clone (example path):

```bash
pip install -e ~/ml/exllamav2
```

Alternatively, the TUI can try importing from a repo path (see EXL2 config key `exl2_repo_path`), but installing editable is usually
simpler.

## Clearing stale extension builds

If you change CUDA toolkits / torch versions, clear cached builds:

```bash
rm -rf ~/.cache/torch_extensions/*exllamav2*
```

Then rerun the import (it will rebuild).

## Resolving CUDA version mismatch

If builds fail with CUDA errors, focus on aligning these:
- `torch.version.cuda`
- `nvcc --version`
- `CUDA_HOME` (where your toolkit lives)

Two practical paths:

1) **Install a CUDA toolkit matching your PyTorch build** (preferred).
   - If `torch.version.cuda` is `12.8`, install a CUDA 12.8 toolkit and ensure `nvcc` resolves to it.

2) **Install a PyTorch build matching your existing CUDA toolkit**.
   - If your `nvcc` is `12.4`, use a PyTorch wheel built for CUDA 12.4 (if available for your torch version).

Notes:
- Under WSL, it’s common to have multiple toolkits installed. Verify which one `nvcc` points to.
- ExLlamaV2 builds against the toolkit pointed to by `nvcc`/`CUDA_HOME`, while runtime uses the CUDA libraries shipped with your torch wheel.

