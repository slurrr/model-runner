# Primary dev machine

Captured: 2026-03-02

## Hardware (user-reported)
- GPU: NVIDIA RTX 4090 (24 GB VRAM)
  - Note: user reported “55.8 GB of GPU memory” shown by the OS/UI; that typically includes **shared/system memory**. This repo’s guidance assumes **VRAM (24 GB)** when discussing “GPU memory”.
- CPU: Intel i9
- RAM: 64 GB

## Usage intent
- Prefer running models fully on VRAM (no CPU offload).
- Used for local HF checkpoints under `/home/poop/ml/models/` and for evaluating long-context behavior.

