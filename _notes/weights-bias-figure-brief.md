# Weights/Bias Figure Brief

## Concept

- **Topic:** Weights and bias in a neural-network unit
- **Audience:** Curious beginners who can read simple charts but do not yet trust the jargon
- **Single lesson:** Weights determine which inputs pull the score up or down; bias shifts when the unit fires without changing which inputs matter most.

## After Viewing, The Reader Should Be Able To Answer

1. Which inputs are pulling the score most strongly, and in which direction?
2. What stays fixed when only the bias changes?
3. How does changing the bias move the threshold or decision rule?

## Structure

- **Entities / parts:** inputs, weights, per-feature contributions, bias, total score, activation threshold, decision boundary
- **Variables / symbols:** `x_i`, `w_i`, `b`, `z = \sum_i w_i x_i + b`, `\sigma(z)`
- **What changes:** bias value in the dynamic view; representation of the same rule across variants
- **What stays fixed:** the input pattern and the weights when demonstrating threshold shift
- **How the parts influence one another:** each contribution is `w_i x_i`; the bias offsets the total score; the activation maps that total score to a response
- **Common confusion or misconception:** people often think bias is just “another weight” instead of a global shift, and many graphics hide sign/magnitude so the score never feels concrete

## Candidate Variants

### Variant A — Structural
- Best for: first-time readers who need labeled parts and a stable mental model
- Shows clearly: what the pieces are and the order of operations
- Still weak on: exact magnitude and sign of influence

### Variant B — Quantitative
- Best for: readers who want to see how the score is literally built
- Shows clearly: input values, signed weights, contributions, total score, threshold shift
- Still weak on: intuition for boundary movement in feature space

### Variant C — Dynamic
- Best for: readers confused specifically about what bias changes
- Shows clearly: same weights, moving threshold, parallel decision-rule shift
- Still weak on: static part labels unless paired with another panel

## Chosen Direction

- **Selected variant or hybrid:** not chosen yet; produce A/B/C in one comparison board first
- **Why:** the user explicitly wants to compare permutations before committing
- **Tooling plan:** `matplotlib` for all three variants, one board PNG, optional GIF frame references for dynamic variant
- **Caption plan:** keep in-image text short, put comparative explanation beneath or alongside the board on a preview page
