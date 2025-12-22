# AI Tool Usage Log

AI tools were used for debugging and understanding, not to fabricate results.

## exp0_baseline
- Interpreting CorrDiff pipeline stages (regression vs diffusion vs generation)
- Debugging import error for InfiniteSampler (API path change)
- Debugging diffusion checkpoint loader expectations (.mdlus vs .pt)
- Reasoning about CPU-safe config reductions (batch size, training_duration, workers)

All changes were validated by rerunning the pipeline and inspecting outputs.
