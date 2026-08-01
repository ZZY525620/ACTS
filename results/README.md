# Result Tables

This directory contains lightweight CSV summaries used for manuscript tables.

## FLARE22 (test cases 0041-0050)

- `flare_full_dice_summary.csv`: FLARE22 full 7-method Dice comparison on cases 0041-0050.
- `flare_full_nsd_summary.csv`: FLARE22 full 7-method NSD comparison.
- `flare_module_ablation_dice.csv`: FLARE22 four-organ main module ablation (Initial -> Rule -> DQN -> SAM Adaptation -> Ours).
- `flare_liver_sam_adaptation_ablation.csv`: FLARE22 liver internal ablation for SAM adaptation strategies (augweighted vs augonly vs rewardonly).

## AMOS22 (test cases 0014-0015, equivalent to 2-case protocol)

- `amos_full_dice_summary.csv`: AMOS22 full 7-method Dice comparison. Ours is FT-DQN trained on AMOS train set.
- `amos_full_nsd_summary.csv`: AMOS22 full 7-method NSD comparison.
- `amos_fixed_dsc_summary.csv`: AMOS22 frozen/fine-tuned ACTS Dice summary for 3-case and 2-case test sets.
- `amos_fixed_dsc_percase.csv`: per-case AMOS22 Dice values for frozen and fine-tuned ACTS.

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
