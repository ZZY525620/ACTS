# Experiment Protocol

This document records the dataset splits, organ labels, and main settings used for the current ACTS experiments.

## Datasets

The repository does not redistribute datasets. FLARE22 and AMOS22 should be downloaded from their official sources and converted to the local layout described in the root README.

## Organ Labels

| Dataset | Liver | Spleen | Right kidney | Left kidney |
|---|---:|---:|---:|---:|
| FLARE22 | 1 | 3 | 2 | 13 |
| AMOS22 | 6 | 1 | 2 | 3 |

## FLARE22 Split

For the main large-organ experiments:

- Training cases: `0001`-`0040`
- Test cases: `0041`-`0050`
- Reported organs: liver, spleen, right kidney, left kidney
- Reported metric in the main table: Dice, rounded to four decimals

## AMOS22 Split

For the fixed AMOS pipeline:

- Training cases: `amos_0001`, `amos_0004`, `amos_0005`, `amos_0006`, `amos_0007`, `amos_0009`, `amos_0010`
- Validation/auxiliary test case: `amos_0011`
- Comparison test cases: `amos_0014`, `amos_0015`
- Reported organs: liver, spleen, right kidney, left kidney

The comparison table uses `amos_0014` and `amos_0015` to match the single-case/two-case comparison setting used by the external baselines.

## ACTS Settings

Unless otherwise noted:

- SAM backbone: original Segment Anything ViT-B checkpoint.
- SAM input resolution: `256 x 256` for sequence candidate generation.
- DQN epochs: `40`.
- Maximum DQN steps per episode: `20`.
- SAM adaptation: train only the SAM mask decoder.
- SAM adaptation epochs: `4`.
- Pseudo-label selection: DQN-accepted high-confidence slices.
- Fine-tuning strategies: reward-weighted pseudo-label loss and prompt augmentation.

## Metrics

The shared metric script computes:

- Dice coefficient.
- IoU.
- Normalized surface Dice (NSD).

All methods should be evaluated from binary foreground NIfTI predictions with the same evaluator. For multi-class label files, the evaluator extracts the organ by dataset-specific label.

## Reproducibility Notes

Generated prediction volumes, caches, checkpoints, and visualizations are not tracked by Git. They should be regenerated locally from the scripts and stored under `outputs/`.
