# Scripts

This directory contains reproducible entry points used in the current release.

- `run_amos_full_pipeline_fixed.py`: fixed AMOS pipeline for cache construction, frozen-SAM DQN training, SAM mask-decoder adaptation, cache rebuilding, and final DQN evaluation.
- `evaluate_binary_nii_metrics.py`: shared metric script for binary NIfTI predictions. It computes Dice, IoU, and NSD from a CSV manifest.

Large intermediate outputs are intentionally written outside Git-tracked files, usually under `outputs/`.

FLARE experiments can be reproduced by combining the package entry points:

- `python -m acts.rl.candidate_cache` to build per-case candidate caches.
- `python -m acts.rl.train_dqn_multicase` to train DQN policies from those caches.
- `python -m acts.sam.finetune_mask_decoder_pseudo` to adapt the SAM mask decoder with pseudo labels.
- `python -m acts.rl.evaluate_dqn` or `python -m acts.rl.evaluate_dqn_multicase` to evaluate cached DQN policies.

See [../docs/reproduce.md](../docs/reproduce.md) for concrete commands.
