# Explanatory Figure Quality Rubric

This is the working 50-point rule for article graphics in this repo.

## Scoring

- Score each criterion from `0` to `5`.
- Ten criteria × five points each = `50` total.
- If a criterion is not central to that figure type, give it a `5` only if the figure avoids creating a false problem in that area.

## Ship Rule

- `40/50` is the minimum shipping score.
- No single criterion may score below `3/5`.
- Any visible text collision, cropped label, or unreadable overlap is an automatic fail even if the raw total is above `40`.

## Criteria

1. `Lesson Clarity`
   - Can a viewer say what the figure teaches in one sentence?
2. `Plain-Language Labels`
   - Do labels explain the job first instead of hiding behind jargon?
3. `Reading Order`
   - Is the left-to-right or top-to-bottom scan obvious?
4. `Fixed vs Changing`
   - Is it clear what stays fixed and what changes?
5. `Quantitative Honesty`
   - If magnitude matters, are scales, thresholds, or numeric anchors shown?
6. `Math Alignment`
   - Does the visual match the equations and symbols used in the article?
7. `Text Fit`
   - Are labels readable, uncrowded, uncropped, and collision-free?
8. `Visual Economy`
   - Did we remove decorative junk and keep only meaning-bearing elements?
9. `Accessibility`
   - Is contrast strong enough and color not doing all the work alone?
10. `Editorial Polish`
   - Does it look like a publication figure instead of a slide or AI infographic?

## Fast Triage Rule

- Fix `Text Fit` first.
- Then fix `Lesson Clarity` and `Fixed vs Changing`.
- Only polish color, spacing, and style after those three pass.
