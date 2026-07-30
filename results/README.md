# Result Tables

This directory contains lightweight CSV summaries used for manuscript tables.

- `flare_3method_dice_summary.csv`: FLARE22 test results on cases `0041`-`0050`.
- `flare_module_ablation_dice.csv`: FLARE22 four-organ main module ablation on cases `0041`-`0050`.
- `flare_liver_sam_adaptation_ablation.csv`: FLARE22 liver internal ablation for SAM adaptation strategies.
- `amos_fixed_dsc_summary.csv`: AMOS22 frozen/fine-tuned ACTS Dice summary for `amos_0011`, `amos_0014`, and `amos_0015`, plus the comparison subset `amos_0014` and `amos_0015`.
- `amos_fixed_dsc_percase.csv`: per-case AMOS22 Dice values.

The CSV files are included for transparent table reconstruction only. Full prediction volumes are not included.

## Column Notes

For `flare_module_ablation_dice.csv`:

- `initial_sam`: initial sequence propagated by frozen SAM.
- `rule`: rule-selected candidates on frozen SAM caches.
- `dqn_agent`: DQN-selected candidates on frozen SAM caches.
- `sam_adaptation`: initial sequence after SAM mask-decoder adaptation.
- `sam_adaptation_dqn`: final ACTS result after SAM adaptation, cache rebuilding, and DQN retraining.

For `flare_liver_sam_adaptation_ablation.csv`:

- `prompt_augmentation_reward_weighting`: full SAM adaptation setting.
- `prompt_augmentation_only`: removes reward-weighted pseudo-label loss.
- `reward_weighting_only`: removes prompt augmentation.
