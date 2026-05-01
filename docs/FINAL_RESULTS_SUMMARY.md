# Final Results Summary

## Core Question

Can NLP-derived event signals improve abnormal next-week port activity prediction beyond an operational baseline?

## Final Setup

- Case: Port of Rotterdam
- Operational data: IMF PortWatch weekly port activity
- Event data: GDELT maritime news event records
- Training period: 2021-2024
- Test period: 2025
- Target: abnormal next-week port activity

## Main Result

The strongest result comes from a two-stage multiscale NLP correction model.

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Operational baseline | 0.815 | 0.471 | 0.500 | 0.700 | 0.389 |
| Multiscale NLP correction | 0.838 | 0.505 | 0.593 | 0.800 | 0.471 |

## Threshold Sensitivity

At the best-F1 threshold:

| Model | Threshold | F1 | Recall | Precision |
|---|---:|---:|---:|---:|
| Operational baseline | 0.45 | 0.581 | 0.900 | 0.429 |
| Multiscale NLP correction | 0.45 | 0.600 | 0.900 | 0.450 |

## Interpretation

The results suggest that NLP-derived event signals are useful when they are integrated as spatially layered correction signals. Simple global news features are unstable, while multiscale alignment across global, Europe-level, and local signals improves the operational baseline.

## Main Claim

NLP signals are not automatically useful as raw external variables. Their predictive value improves when they are spatially aligned and used as a correction layer on top of operational risk estimates.
