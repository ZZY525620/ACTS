# ACTS Package

This directory contains the core implementation.

- `data`: NIfTI loading, CT windowing, resizing, and dataset path resolution.
- `prompts`: conversion from masks to box, positive/negative point, mask, and interpolation prompts.
- `sequence`: SAM-based slice propagation, anomaly scoring, and state-aware candidate scoring.
- `rl`: candidate-cache construction, DQN environment, DQN policy training, and evaluation.
- `sam`: SAM wrapper and mask-decoder fine-tuning with pseudo labels.
- `evaluation`: Dice/IoU metrics, diagnostics, and visualization helpers.

The package name is `acts`, so modules can be run or imported as:

```python
from acts.sequence.propagate import propagate_sequence
from acts.rl.env import CTSliceAgentEnv
```
