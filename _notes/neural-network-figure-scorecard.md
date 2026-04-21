# Neural Network Article Figure Scorecard

Legend:

- `LC` lesson clarity
- `PL` plain-language labels
- `RO` reading order
- `FC` fixed vs changing
- `QH` quantitative honesty
- `MA` math alignment
- `TF` text fit
- `VE` visual economy
- `AC` accessibility
- `EP` editorial polish

Shipping threshold: `40/50` and no score below `3`.

| Figure | LC | PL | RO | FC | QH | MA | TF | VE | AC | EP | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `explanation-problem.png` | 4 | 5 | 5 | 4 | 5 | 4 | 5 | 4 | 4 | 4 | 44 |
| `neuron-scoring-rule.png` | 5 | 4 | 5 | 4 | 5 | 5 | 4 | 4 | 4 | 4 | 44 |
| `weights-biases.png` | 5 | 4 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 46 |
| `weights-bias-threshold-shift.gif` | 5 | 4 | 5 | 5 | 5 | 4 | 4 | 4 | 4 | 4 | 44 |
| `activation-functions.png` | 4 | 4 | 5 | 4 | 5 | 5 | 5 | 4 | 4 | 4 | 44 |
| `representation-building.png` | 4 | 4 | 5 | 5 | 4 | 4 | 5 | 4 | 4 | 4 | 43 |
| `backprop-blame-assignment.png` | 5 | 4 | 4 | 5 | 4 | 5 | 4 | 4 | 4 | 4 | 43 |
| `debugging-checklist.png` | 4 | 5 | 5 | 4 | 5 | 4 | 5 | 3 | 4 | 4 | 43 |

## Notes By Figure

- `explanation-problem.png`
  - Clear and clean now.
  - Still a little box-heavy, but not confusing.

- `neuron-scoring-rule.png`
  - Good teaching arc from inputs to weighted sum to activation.
  - Next improvement would be slightly more breathing room around the left table.

- `weights-biases.png`
  - Strongest current figure because the example is concrete and the cutoff shift is explicit.
  - Keep using this style: one scenario, one score, one decision difference.

- `weights-bias-threshold-shift.gif`
  - Much cleaner after the collision fix.
  - Still a bit dense in the left summary block, so it should be watched in future passes.

- `activation-functions.png`
  - The title collision is fixed.
  - Still more generic than the best figures in the set because it shows curves but not a concrete downstream consequence.

- `representation-building.png`
  - Good before/after progression.
  - Could get stronger with one more explicit cue about why the final line reads the later representation better.

- `backprop-blame-assignment.png`
  - The story is there now: forward pass, loss, backward signal, gradient sizes.
  - The lower bar section is still busier than it should be.

- `debugging-checklist.png`
  - Legible and useful, but the most slide-like figure in the set.
  - If this gets another pass, reduce the box treatment and make the hierarchy more editorial.
