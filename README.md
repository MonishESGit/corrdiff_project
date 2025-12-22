# CorrDiff HRRR-Mini Tiny Project (NCSU GIC Lab)

This repo documents my tiny project submission for reproducing the CorrDiff pipeline on the HRRR-Mini dataset using NVIDIA PhysicsNeMo.

The work is organized as **experiments** under `experiments/`, each containing:
- configs used
- exact commands (runbook)
- small verification artifacts (metrics, NetCDF header)
- brief notes

## Part 1 (Completed): End-to-end reproduction on local CPU
Baseline experiment: `experiments/exp0_baseline/`

Completed stages:
1. Regression training (CPU smoke test)
2. Diffusion training conditioned on regression checkpoint (CPU smoke test)
3. Generation to NetCDF output + basic verification

## Part 2 (In progress): Architecture + deeper understanding
I will add notes on:
- model architecture (regression UNet, diffusion residual model)
- how stages interact (conditioning, residuals)
- loss functions and training objectives
- additional controlled experiments

Planned notes will live in `report/report_part2_architecture.md` and experiment folders.

---

## Environment
- Machine: MacBook Pro (Apple Silicon)
- Compute: CPU-only
- Python: 3.11
- W&B: offline

---

## Where to start (for reviewers)
Open:
- `experiments/exp0_baseline/runbook.md`
- `report/report_part1_baseline.md`
- `experiments/exp0_baseline/results/ncdump_header.txt`

---

## Repo layout
- `experiments/`: each experiment is self-contained
- `scripts/`: small utilities for metrics/inspection
- `report/`: Part 1 report + Part 2 notes placeholder
- `physicsnemo/`: cloned dependency (or submodule)
- `ai_usage.md`: record of AI-tool usage
