# Scripts

This directory contains reproducible entry points used in the current release.

- `run_amos_full_pipeline_fixed.py`: fixed AMOS pipeline for cache construction, frozen-SAM DQN training, SAM mask-decoder adaptation, cache rebuilding, and final DQN evaluation.
- `evaluate_binary_nii_metrics.py`: shared metric script for binary NIfTI predictions. It computes Dice, IoU, and NSD from a CSV manifest.

Large intermediate outputs are intentionally written outside Git-tracked files, usually under `outputs/`.
