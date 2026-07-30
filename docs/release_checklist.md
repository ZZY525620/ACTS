# Release Checklist

This checklist records what is included in the anonymous ACTS repository and what is intentionally excluded.

## Included

- Core ACTS Python package under `acts/`.
- Candidate pool generation code.
- Sequence propagation and abnormal-slice selection code.
- DQN state/action/reward environment and training code.
- SAM wrapper and mask-decoder adaptation code.
- FLARE22 and AMOS22 pipeline entry scripts.
- Shared Dice/IoU/NSD evaluator.
- Experiment protocol documentation.
- Candidate pool documentation.
- Reproduction guide.
- Lightweight CSV result summaries.

## Excluded

- Medical image data.
- Ground-truth NIfTI label files.
- Generated prediction NIfTI files.
- Candidate cache `.npz` files.
- SAM checkpoints.
- Fine-tuned SAM model weights.
- DQN model weights.
- Large visualizations.
- Local experiment folders.
- Private notes and local absolute paths.

## Pre-Submission Checks

Before sharing a new release commit:

```bash
python -m compileall -q acts scripts
python -c "import acts; import acts.rl.candidate_cache; import acts.sam.sam_tool; print('import OK')"
```

Recommended sensitive-text scan:

```bash
rg -n "ABSOLUTE_LOCAL_PATH|PRIVATE_USER_NAME|PRIVATE_PROJECT_NAME|sk-[A-Za-z0-9_-]{8,}" -S .
```

The scan should not report local paths, personal identifiers, API keys, or private notes.

## Current Release Status

The current repository is suitable as an anonymous review code release. It contains the method implementation, scripts, protocols, and compact result tables, while excluding data, weights, and large generated outputs.
