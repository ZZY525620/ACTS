# Reproduction Guide

This guide describes the expected local setup and the main commands for reproducing ACTS experiments. The repository does not include medical images, annotations, model checkpoints, generated predictions, or caches.

## Environment

Install PyTorch following the official instructions for your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

The original Segment Anything ViT-B checkpoint should be available as:

```text
sam_vit_b_01ec64.pth
```

## Data

Prepare datasets as:

```text
Data/
+-- FLARE22/
|   +-- images/
|   |   +-- FLARE22_Tr_0001_0000.nii.gz
|   +-- labels/
|       +-- FLARE22_Tr_0001.nii.gz
+-- amos/
    +-- images/
    |   +-- amos_0001.nii.gz
    +-- label/
        +-- amos_0001.nii.gz
```

## Single-Sequence Smoke Test

Run one FLARE liver sequence:

```bash
python -m acts.main_liver_mvp \
  --data-dir Data/FLARE22 \
  --case-id 0001 \
  --liver-label 1 \
  --model-path sam_vit_b_01ec64.pth \
  --output-dir outputs/smoke_flare0001_liver \
  --device cuda
```

## Candidate Cache

Build a candidate cache for one FLARE case and organ:

```bash
python -m acts.rl.candidate_cache \
  --data-dir Data/FLARE22 \
  --case-id 0001 \
  --liver-label 1 \
  --output-dir outputs/cache_flare0001_liver \
  --model-path sam_vit_b_01ec64.pth \
  --topk-ratio 0.5 \
  --fp-aware \
  --device cuda
```

For other FLARE organs, change `--liver-label` according to the label table:

| Organ | FLARE22 label |
|---|---:|
| liver | 1 |
| right_kidney | 2 |
| spleen | 3 |
| left_kidney | 13 |

## Train DQN from Caches

After building caches for multiple training cases, train a DQN policy:

```bash
python -m acts.rl.train_dqn_multicase \
  --train-cache-dirs outputs/cache_flare0001_liver outputs/cache_flare0002_liver \
  --eval-cache-dirs outputs/cache_flare0041_liver outputs/cache_flare0042_liver \
  --output-dir outputs/dqn_flare_liver \
  --epochs 40 \
  --max-steps 20 \
  --device cuda
```

## SAM Mask-Decoder Adaptation

Fine-tune only the SAM mask decoder with DQN pseudo labels:

```bash
python -m acts.sam.finetune_mask_decoder_pseudo \
  --data-dir Data/FLARE22 \
  --case-ids 0001 0002 0003 \
  --pseudo-root outputs/pseudo_labels_liver \
  --output-dir outputs/sam_adapt_liver \
  --model-path sam_vit_b_01ec64.pth \
  --liver-label 1 \
  --prompt-mode box_point_mask \
  --augmentations-per-sample 3 \
  --box-jitter-ratio 0.08 \
  --point-mode random \
  --pseudo-weight-mode reward \
  --epochs 4 \
  --device cuda
```

Then rebuild candidate caches with the adapted SAM checkpoint and retrain/evaluate the DQN policy.

## Fixed AMOS Pipeline

The AMOS pipeline script runs the complete sequence:

1. Build frozen-SAM caches.
2. Train frozen-cache DQN per organ.
3. Evaluate frozen DQN.
4. Generate DQN pseudo labels.
5. Fine-tune SAM mask decoder per organ.
6. Rebuild caches with adapted SAM.
7. Train adapted-cache DQN per organ.
8. Evaluate adapted DQN.

```bash
python scripts/run_amos_full_pipeline_fixed.py \
  --data-dir Data/amos \
  --sam-checkpoint sam_vit_b_01ec64.pth \
  --organs liver spleen right_kidney left_kidney \
  --output-dir outputs/amos_pipeline_fixed \
  --dqn-epochs 40 \
  --max-steps 20 \
  --ft-epochs 4 \
  --device cuda
```

## Metrics

Use the shared evaluator for all methods:

```bash
python scripts/evaluate_binary_nii_metrics.py \
  --manifest manifest.csv \
  --out metrics.csv \
  --summary-out summary.csv \
  --tolerance-mm 1.0
```

The manifest should contain:

```text
method,dataset,case_id,organ,pred_path,gt_path,label
```

For binary prediction files, `pred_path` should point to a 0/1 NIfTI mask. For multi-class GT files, `label` can be provided explicitly or inferred from the dataset/organ table.

## Reported Splits

FLARE22:

- Train: `0001`-`0040`
- Test: `0041`-`0050`

AMOS22:

- Train: `amos_0001`, `amos_0004`, `amos_0005`, `amos_0006`, `amos_0007`, `amos_0009`, `amos_0010`
- Comparison test subset: `amos_0014`, `amos_0015`

See [experiment_protocol.md](experiment_protocol.md) for the exact settings used in the reported tables.

## FLARE22 Full Pipeline

The FLARE22 large-organ pipeline uses the same ACTS stages as the AMOS script, with FLARE-specific case naming and labels:

```bash
python scripts/run_flare_full_pipeline.py \
  --data-dir Data/FLARE22 \
  --sam-checkpoint sam_vit_b_01ec64.pth \
  --organs liver spleen right_kidney left_kidney \
  --output-dir outputs/flare_pipeline \
  --dqn-epochs 40 \
  --max-steps 20 \
  --ft-epochs 4 \
  --device cuda
```

Use `--skip-sam-ft` to run only frozen-SAM cache construction, frozen-cache DQN training, and frozen DQN evaluation.
