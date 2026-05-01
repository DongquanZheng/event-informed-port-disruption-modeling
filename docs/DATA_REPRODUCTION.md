# Data Reproduction Guide

This project does not redistribute raw PortWatch or GDELT data files. The raw files can be large and should be obtained directly from the original public sources. Instead, the repository provides a clean reproduction script that documents the full data pipeline.

## Reproduction Levels

### 1. Lightweight reproduction

Use this option if you only want to reproduce the final result tables and figures from the small summary files committed to the repository.

```bash
jupyter notebook notebooks/final_multiscale_model_summary.ipynb
```

This is the recommended path for quick GitHub review.

### 2. Full pipeline reproduction

Use this option if you want to rebuild the dataset from public sources.

```bash
python scripts/reproduce_pipeline.py
```

The script performs the following steps:

1. Downloads Rotterdam daily port activity from the PortWatch ArcGIS REST API.
2. Aggregates daily PortWatch records into weekly port activity.
3. Constructs the abnormal next-week activity target.
4. Downloads daily GDELT Event Database files for the selected years.
5. Filters maritime and logistics-related GDELT records.
6. Extracts URL slug text from `SOURCEURL`.
7. Builds weak labels from event severity, tone, and disruption-related keywords.
8. Trains a weakly supervised TF-IDF Logistic Regression NLP model.
9. Scores article-level disruption probabilities.
10. Aggregates NLP event signals into weekly global, Europe-level, and local features.
11. Trains an operational baseline and a two-stage multiscale NLP correction model.
12. Saves reproduced processed files under `data/processed/`.

Detailed NLP filtering, weak-labeling, spatial-layer construction, and feature aggregation rules are documented in:

```text
docs/NLP_SIGNAL_CONSTRUCTION.md
```

## Runtime Note

Full reproduction can take a long time because GDELT daily event files must be downloaded and processed for multiple years. For a faster test, run the script with fewer years:

```bash
python scripts/reproduce_pipeline.py --years 2024 2025
```

The final reported result in the README uses 2021-2024 as the training period and 2025 as the test period.

## Data Source Notes

PortWatch data are accessed through the public PortWatch ArcGIS REST API.

GDELT data are accessed through public daily event files using URLs of the form:

```text
http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
```

The project uses the GDELT `SOURCEURL` field as a lightweight text representation. Full news article text is not redistributed.

## Expected Differences

The full reproduction script is designed to reproduce the methodology rather than freeze an exact historical artifact. Small differences may occur if public source data are updated, unavailable for some dates, or if GDELT records change across downloads.

For exact figure/table reproduction, use the lightweight summary files already included in `data/processed/`.
