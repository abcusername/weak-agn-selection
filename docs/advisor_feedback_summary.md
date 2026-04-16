# Advisor Feedback Summary

## Main Issues Identified

1. Some spectra are off-center, which may lead to double counting and incorrect classification.
2. High Fvar alone is not sufficient to identify AGN activity.
3. Outlier treatment should be improved, because some outliers may contain real variability signals.
4. Multi-band correlation should be checked, since real AGN variability is expected to be correlated across bands.
5. Low-variability Seyferts should also be examined to understand the limitations of variability selection.

## Current Response in This Repository

The current workflow has already started addressing these issues through:

- central-source uniqueness in the MINFIX cleaning pipeline
- noise-control and magnitude-dependent baseline analysis
- QC inspection of bright Seyfert control samples
- high-Fvar case studies
- multiband and center-check follow-up scripts

## Next Planned Steps

- stricter center-matching
- multi-band variability comparison
- improved variability metrics beyond fixed Fvar
- validation of clean high-variability candidates
