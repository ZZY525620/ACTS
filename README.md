# ACTS

ACTS is a CT sequence segmentation framework built around Segment Anything (SAM), sequence prompt propagation, candidate-mask correction, and a lightweight DQN policy. The repository is prepared for anonymous review and contains code, experiment protocols, and compact result tables. Medical images, annotations, generated NIfTI predictions, and model checkpoints are not included.

## Method Overview

ACTS follows a slice-sequence workflow:

1. Select one reference slice and use its ground-truth mask as the seed prompt.
2. Generate an initial 3D mask by propagating SAM prompts forward and backward through the CT sequence.
3. Detect uncertain slices and build a candidate pool from box, point, mask, interpolation, and fallback prompts.
4. Train a DQN policy to choose candidate masks that improve the sequence result.
5. Optionally adapt only the SAM mask decoder with high-confidence pseudo labels using reward weighting and prompt augmentation.

## Repository Structure

```text
.
+-- acts/                         # Core ACTS package
|   +-- data/                     # NIfTI loading and CT preprocessing
|   +-- prompts/                  # Prompt generation from masks
|   +-- sequence/                 # Sequence propagation and scoring
|   +-- rl/                       # DQN environment, cache, training, evaluation
|   +-- sam/                      # SAM wrapper and mask-decoder adaptation
|   +-- evaluation/               # Metrics and visualization utilities
+-- scripts/                      # Reproducible experiment/evaluation scripts
+-- docs/                         # Experiment protocol notes
+-- results/                      # Lightweight CSV summaries
+-- requirements.txt
+-- README.md
```

## Installation

Create a Python environment with PyTorch and install the remaining dependencies:

```bash
pip install -r requirements.txt
```

ACTS uses the original SAM ViT-B checkpoint by default. Download `sam_vit_b_01ec64.pth` from the official Segment Anything release and place it in the repository root, or pass its path with `--model-path` when supported by a script.

## Data Layout

The code expects public datasets to be prepared locally. A typical layout is:

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

Dataset labels used in our experiments:

| Dataset | Liver | Spleen | Right kidney | Left kidney |
|---|---:|---:|---:|---:|
| FLARE22 | 1 | 3 | 2 | 13 |
| AMOS22 | 6 | 1 | 2 | 3 |

## Example Commands

Run a single FLARE liver sequence:

```bash
python -m acts.main_liver_mvp \
  --data-dir Data/FLARE22 \
  --case-id 0001 \
  --model-path sam_vit_b_01ec64.pth \
  --device cuda
```

Run the fixed AMOS pipeline for four organs:

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

Run the FLARE22 four-organ pipeline:

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

Evaluate binary NIfTI predictions with Dice, IoU, and NSD:

```bash
python scripts/evaluate_binary_nii_metrics.py \
  --manifest manifest.csv \
  --out metrics.csv \
  --summary-out summary.csv
```

For more details, see:

- [docs/method_overview.md](docs/method_overview.md): ACTS pipeline, DQN state/action/reward, and SAM adaptation.
- [docs/candidate_pool.md](docs/candidate_pool.md): exact candidate pool construction, including box perturbations and point prompts.
- [docs/experiment_protocol.md](docs/experiment_protocol.md): dataset splits, labels, and experiment settings.
- [docs/reproduce.md](docs/reproduce.md): local setup and reproduction commands.

## Current Results

The compact CSV summaries in [results/](results/) report the current frozen/fine-tuned ACTS results and comparison baselines. The main ACTS Dice values are:

| Dataset | Test cases | Liver | Spleen | Right kidney | Left kidney | Mean |
|---|---|---:|---:|---:|---:|---:|
| FLARE22 | 0041-0050 | 0.8449 | 0.9471 | 0.9440 | 0.9637 | 0.9249 |
| AMOS22 | 0014, 0015 | 0.7568 | 0.6920 | 0.8428 | 0.8619 | 0.7884 |

Additional CSV files in [results/](results/) include FLARE module ablations and SAM adaptation internal ablations.

## Notes

- This repository does not include medical images, labels, SAM checkpoints, fine-tuned SAM weights, DQN policies, or generated prediction volumes.
- The release is organized for anonymous review; identifying local paths and private notes have been removed.
- Citation information will be added after publication.
