# Candidate Pool Construction

This document summarizes how the DQN candidate pool is built for each selected abnormal slice.

## Inputs

For a target slice `t`, ACTS uses the current sequence prediction around the slice:

- `prev_mask`: mask from slice `t-1`.
- `cur_mask`: current mask from slice `t`.
- `next_mask`: mask from slice `t+1`.
- `interp`: intersection-style interpolation of `prev_mask` and `next_mask`.

The candidate pool is generated only for selected abnormal slices. Boundary slices without both neighbors are skipped.

## Sequence State

Before prompt generation, ACTS estimates one of six sequence states:

- `stable`
- `expanding`
- `shrinking`
- `disappearing`
- `drift`
- `unreliable`

The state controls which box and mask prompt variants are generated.

## Box Prompt Variants

For each source mask in `{prev_mask, next_mask, interp}`, ACTS first extracts the foreground bounding box with no extra expansion. It then generates state-aware box variants:

| State | Box variants |
|---|---|
| `stable` | `tight_box` = 1.00, `expand_box_5` = 1.05, `shrink_box_10` = 0.90 |
| `expanding` | `tight_box` = 1.00, `expand_box_5` = 1.05, `expand_box_10` = 1.10 |
| `shrinking` | `tight_box` = 1.00, `shrink_box_10` = 0.90, `shrink_box_20` = 0.80 |
| `disappearing` | `tight_box` = 1.00, `shrink_box_10` = 0.90, `shrink_box_20` = 0.80 |
| `unreliable` | `tight_box` = 1.00, `shrink_box_10` = 0.90, `shrink_box_20` = 0.80 |
| `drift` | `tight_box` = 1.00, `expand_box_5` = 1.05 |

For `drift`, `shrinking`, and `disappearing`, ACTS also adds shifted boxes:

- `shifted_box_left`
- `shifted_box_right`
- `shifted_box_up`
- `shifted_box_down`

The shift size is:

```text
max(4 pixels, 0.08 * max(box_width, box_height))
```

All boxes are clipped to image boundaries.

## Box + Point Prompts

For each valid box variant, ACTS can also add a positive point. The point is the core point of the source mask:

```text
core point = foreground pixel with maximum Euclidean distance to the mask boundary
```

This point is deterministic and lies near the most stable internal region of the mask.

ACTS does not add box+point variants when the sequence state is `disappearing` or `unreliable`, because forcing a positive point may preserve a false-positive region.

## Mask Prompts

For each non-empty source mask, ACTS adds mask-prompt variants depending on the sequence state:

| Prompt type | Used when |
|---|---|
| plain mask prompt | `stable`, `drift` |
| eroded mask prompt, 2 iterations | `shrinking`, `disappearing`, `unreliable`, `stable` |
| dilated mask prompt, 2 iterations | `expanding`, `stable` |

The mask prompt is converted to SAM low-resolution logits with value `20` inside the mask and `-20` outside.

## Positive Missing-Region Point

ACTS identifies potential false-negative regions by:

```text
miss = (interp > 0) and (cur_mask == 0)
```

If `miss` is non-empty, ACTS extracts its core point and adds:

```text
positive_miss_point
```

This prompt asks SAM to recover regions supported by neighboring slices but missed in the current slice.

## Negative False-Positive Point

ACTS identifies potential false-positive regions by:

```text
fp = (cur_mask > 0) and (interp == 0)
```

If `fp` is non-empty, ACTS extracts its core point and adds:

```text
negative_fp_point
```

This prompt uses label `0` and also provides `interp` as a mask input, so SAM receives both a negative point and a neighboring-slice shape prior.

## SAM Candidate Expansion

Each SAM prompt is sent to `SamPredictor.predict` with:

```text
multimask_output=True
```

Therefore, each prompt can contribute multiple SAM masks. The candidate name records the prompt name and SAM output index, for example:

```text
prev_tight_box:0
prev_tight_box_point:1
interp_eroded_mask_prompt:2
positive_miss_point:0
negative_fp_point:1
```

## Non-SAM Fallback Candidates

ACTS also appends non-SAM candidates:

- `keep_current_mask`
- `empty_mask`
- `eroded_current_mask`, 2 erosion iterations
- `dilated_current_mask`, 1 dilation iteration
- `interpolated_mask`
- `stop_direction`

These fallback candidates allow the DQN to keep a good current prediction, clear a false-positive slice, or use a conservative interpolation result.

## Candidate-to-Action Mapping

Each candidate is mapped to one or more discrete DQN actions based on its name. For each action, the cached metadata stores the best candidate of that action type according to the state-aware no-GT score. The DQN therefore selects a candidate type, and ACTS executes the best cached mask for that type.

The candidate pool also stores:

- SAM score.
- State-aware no-GT score.
- Candidate area.
- Candidate Dice to GT for training/reward analysis.
- Dice improvement over the current mask.
- Rule candidate index.
- Oracle candidate index.

The oracle candidate is used only to estimate candidate-pool upper bound and is not available during test-time policy selection.
