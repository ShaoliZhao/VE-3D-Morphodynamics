# Suggested experiments

## Required ablations

Run four correspondence variants: expression only, geometry only, equal fusion, and a tuned fusion weight. Report sensitivity to the neighborhood size and Sinkhorn regularization.

## Null controls

Randomly rotate and translate one snapshot. Predictions should be stable. Shuffle expression rows within each snapshot to quantify how much matching depends on molecular identity. Replace the learned coupling with an independent uniform coupling as a lower bound.

## Interpolation evaluation

Train or fit on stages flanking a held-out stage. Compare predicted and observed distributions with the official Task 2 evaluation when available. Add scale error, radial-distribution error, neighborhood-distance error, and cell-type proportion error as interpretable diagnostics.

## Extrapolation evaluation

Use rolling-origin validation: hide the latest available training stage, predict it from earlier stages, and only then select hyperparameters. Report uncertainty across embryos or biological replicates when those identifiers are available.

## Comparators

- Linear interpolation after global rigid alignment.
- Expression-only entropic transport.
- A neural ODE or manifold flow model such as MIOFlow.
- A conditional equivariant graph network trained to predict displacements.
- A conditional generative transport model such as GENOT or flow matching.

The baseline is useful when these methods are evaluated under the same stage splits and invariance tests.
