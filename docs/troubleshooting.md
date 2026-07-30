# Troubleshooting

## SAM Checkpoint Not Found

ACTS does not include SAM checkpoints. Download the original SAM ViT-B checkpoint and either place it at:

```text
sam_vit_b_01ec64.pth
```

or pass the path explicitly:

```bash
python scripts/run_flare_full_pipeline.py --sam-checkpoint /path/to/sam_vit_b_01ec64.pth
```

## Data Path Error

Check that the local data layout matches the README. For FLARE22, ACTS expects names similar to:

```text
Data/FLARE22/images/FLARE22_Tr_0001_0000.nii.gz
Data/FLARE22/labels/FLARE22_Tr_0001.nii.gz
```

For AMOS22, ACTS expects:

```text
Data/amos/images/amos_0001.nii.gz
Data/amos/label/amos_0001.nii.gz
```

## CUDA Is Not Used

Check PyTorch CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If it prints `False`, install a CUDA-enabled PyTorch build that matches the local NVIDIA driver.

## NumPy and Matplotlib Import Error

Some older compiled dependencies may fail with NumPy 2.x. This repository pins:

```text
numpy<2
```

If an environment already has NumPy 2.x installed, downgrade it or recreate the environment before running visualization code.

## Torchvision CUDA Version Warning

Some Windows environments report a `torchvision.extension` CUDA version warning even when the main PyTorch CUDA runtime works. The pipeline scripts include a small guard for this known warning. If training still fails, verify that `torch`, `torchvision`, and the CUDA runtime are installed from compatible builds.

## Missing Generated Files

The repository intentionally excludes:

- NIfTI images and labels.
- Candidate caches.
- Prediction volumes.
- SAM checkpoints.
- Fine-tuned SAM weights.
- DQN policy weights.
- Large visualization folders.

These files should be regenerated locally under `outputs/`.

## Candidate Cache Has No Samples

This can happen when:

- The target organ label is wrong.
- The target organ is absent in the selected case.
- The initial SAM propagation produced no valid foreground.
- `--topk-ratio` or abnormal-slice selection is too restrictive.

First verify the organ label table in [experiment_protocol.md](experiment_protocol.md), then run a single-case smoke test before launching the full pipeline.

## AMOS or FLARE Results Differ Slightly

Small differences can come from GPU nondeterminism, PyTorch/CUDA versions, SAM checkpoint variants, or data preprocessing differences. The reported tables use the splits and settings in [experiment_protocol.md](experiment_protocol.md).
