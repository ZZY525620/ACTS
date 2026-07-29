# SAM-CT Sequence Agent

This repository contains the official implementation placeholder for a CT sequence segmentation agent based on Segment Anything and slice-wise policy refinement.

> The full source code, trained model checkpoints, and detailed running instructions will be organized and released in this repository.

## Overview

The method follows a sequence-level refinement pipeline:

1. Generate an initial 3D organ mask from a reference slice prompt.
2. Propagate masks along the CT sequence with SAM-based prompts.
3. Build a candidate mask pool for uncertain slices.
4. Train a lightweight DQN policy to select improved candidate masks.
5. Optionally adapt the SAM mask decoder using high-confidence pseudo labels.

## Current Status

This is a pre-release placeholder prepared for anonymous review. Code and documentation will be progressively added.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   └── README.md
└── sam_ct_agent/
    └── README.md
```

## Data

Public CT datasets should be downloaded from their official sources. This repository does not include medical images, annotations, generated predictions, or model checkpoints.

## Checkpoints

Large files such as SAM checkpoints, fine-tuned weights, DQN policies, and nnU-Net models are not included in the repository.

## Citation

Citation information will be added after publication.

