# exp2_seed_repro — Runbook

## Objective
Verify reproducibility and controlled stochasticity in CorrDiff generation by fixing and varying the random seed.

---

## Setup
- `num_ensembles = 1`
- Same regression and diffusion checkpoints
- Generation run three times with different seeds

---

## Runs

| Run | Seed | Expected Behavior |
|----|------|------------------|
| Run A | 0 | Reference output |
| Run B | 0 | Identical to Run A |
| Run C | 1 | Different stochastic sample |

---

## Commands

```
python generate.py --config-name=generate_seed0.yaml ++generation.seed=0
```
```
python generate.py --config-name=generate_seed0.yaml ++generation.seed=0```
```
```
python generate.py --config-name=generate_seed0.yaml ++generation.seed=1
```