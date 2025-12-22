# Tiny Project Report (Part 1): CorrDiff HRRR-Mini Reproduction

## Project Duration
~2 days total:
- Environment setup and dependency installation
- Regression training smoke test (CPU)
- Diffusion training smoke test (CPU)
- Generation to NetCDF and verification
- Debugging and documentation

## Learning Outcomes
- Understood CorrDiff’s two-stage design:
  - regression predicts a deterministic mean forecast
  - diffusion models residual uncertainty conditioned on the regression output
- Learned how to run and validate a research pipeline end-to-end under limited compute
- Gained practical experience debugging API mismatches and checkpoint format expectations in PhysicsNeMo

## Challenges
- **API mismatch**: `InfiniteSampler` import path changed; fixed by updating import to `physicsnemo.models.diffusion.training_utils`.
- **Checkpoint format**: diffusion required regression checkpoint as a `.mdlus` artifact (module state) rather than the `.pt` training checkpoint.
- **Resource constraints**: default configs assumed A100 GPUs; adapted training duration and batch sizing for CPU smoke tests.

## Results
Baseline experiment (`exp00_baseline_smoke`) produced:
- Regression checkpoints: `.mdlus`, `.pt`
- Diffusion checkpoints: `.mdlus`, `.pt`
- Generated NetCDF output containing groups: `input`, `truth`, `prediction`

Sanity check metric (10m zonal wind):
- MAE ≈ 2.1556 m/s
- RMSE ≈ 2.7167 m/s

These metrics are used to confirm pipeline correctness and output scaling, not to claim model quality.

## Use of AI Tools
AI tools were used to help interpret errors, understand configuration structure, and reason about the pipeline design. All fixes were verified manually and documented (see `ai_usage_log.md`).

## Notes / Limitations
- CPU-only execution
- reduced training duration for smoke testing
- reduced resolution in generated output (64×64)
