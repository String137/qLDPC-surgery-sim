# qLDPC-surgery-sim

Python code for simulating surgery on CSS qLDPC codes.

- **Build codes:** hypergraph product (HGP), lifted product (LP), bivariate bicycle (BB), and surface codes.
- **Merge patches:** surgery between code patches, including high-rate surgery between HGP codes ([arXiv:2510.08523](https://arxiv.org/abs/2510.08523)).
- **Analyze faults:** phenomenological-noise circuits in Stim, and a search for the minimum-weight fault that flips a logical observable without triggering any detector.

## Layout

```
core/       Qubit, Check, CSS code classes, code patches
surgery/    patch merging and circuit scheduling (pre/merge/split/post)
analysis/   logical fault analysis
viz/        visualization
```

## Installation

## Usage
