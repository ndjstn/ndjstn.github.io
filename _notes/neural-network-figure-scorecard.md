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
| `activation-functions.png` | 4 | 4 | 5 | 4 | 5 | 5 | 5 | 4 | 4 | 5 | 45 |
| `representation-building.png` | 4 | 4 | 5 | 5 | 4 | 4 | 5 | 4 | 4 | 4 | 43 |
| `backprop-blame-assignment.png` | 5 | 4 | 5 | 5 | 4 | 5 | 4 | 4 | 4 | 5 | 45 |
| `debugging-checklist.png` | 4 | 5 | 5 | 4 | 5 | 4 | 5 | 4 | 4 | 5 | 45 |

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
  - Frame-level collisions are cleaned up enough to ship.
  - Still slightly dense on the left summary block, so it should be watched in future passes.

- `activation-functions.png`
  - The title collision is fixed.
  - Still more generic than the best figures in the set because it shows curves but not a concrete downstream consequence.

- `representation-building.png`
  - Good before/after progression.
  - Could get stronger with one more explicit cue about why the final line reads the later representation better.

- `backprop-blame-assignment.png`
  - Better split now: left side explains the signal flow, right side explains which parameters move most.
  - The next polish pass should focus on spacing and typography, not structure.

- `debugging-checklist.png`
  - Much less slide-like after the redesign.
  - This is now a clean editorial checklist instead of a stack of colored boxes.
