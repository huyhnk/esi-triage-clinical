# ESI Triage Prediction from ED Clinical Data

This repository is a cleaned, GitHub-ready version of the exploratory Colab notebook `work.ipynb`.
It focuses on predicting Emergency Severity Index (ESI) / ED `acuity` from structured vital-sign features and free-text chief complaints.

## Main experiments

The repository preserves the final experiment flow used in the notebook:

- Build a master ED table from `edstays`, `triage`, `vitalsign`, `diagnosis`, and `medrecon`.
- Clean ESI/acuity labels and numeric vital-sign variables.
- Convert the original 5 ESI levels into a 3-class task: class 1, class 2, and class 3 = original levels 3/4/5.
- Train numeric baselines using Random Forest and XGBoost.
- Train text-enhanced models by concatenating 10 numeric variables with TF-IDF features from `chiefcomplaint`.
- Save metrics, confusion matrices, trained models, a correlation figure, and Random-Forest feature importance.

## Repository structure

```text
esi-triage-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── run_pipeline.py
├── notebooks/
│   └── work_original.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preparation.py
│   ├── evaluation.py
│   ├── train_numeric.py
│   ├── train_multimodal.py
│   └── analysis_figures.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── outputs/
    ├── figures/
    └── results/
```

## Data

The original notebook uses MIMIC-IV-ED-derived CSV files. These data are **not included** in this repository and should not be committed to GitHub.

Place the following files in `data/raw/`:

```text
edstays.csv
triage.csv
vitalsign.csv
diagnosis.csv
medrecon.csv
```

The preprocessing script also accepts the nested layout used in parts of the original Colab notebook, for example `data/raw/edstays.csv/edstays.csv`.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the complete pipeline

```bash
python run_pipeline.py
```

Or run each stage separately:

```bash
python -m src.data_preparation
python -m src.train_numeric
python -m src.train_multimodal
python -m src.analysis_figures
```

## Outputs

Generated files are written to:

- `data/processed/`: `master_dataset.csv`, `master_dataset_clean.csv`
- `models/`: fitted preprocessing objects and trained models
- `outputs/results/`: CSV model-comparison tables
- `outputs/figures/`: confusion matrices, correlation matrix, feature importance

These generated artifacts are ignored by Git by default.

## Features used in the final numeric experiments

The 10 variables retained in the later notebook experiments are:

1. `vitals_count`
2. `heartrate`
3. `sbp`
4. `dbp`
5. `pain`
6. `resprate_max`
7. `o2sat`
8. `o2sat_min`
9. `heartrate_max`
10. `pain_min`

For the text-enhanced experiment, TF-IDF features are extracted from `chiefcomplaint` using up to 3,000 unigram/bigram features with `min_df=5`.

## Reproducibility notes

The exploratory notebook contains repeated Colab mounts, repeated experiments, and several intermediate model versions. The `src/` scripts consolidate the later, more stable 3-class numeric and numeric+text experiments into a reproducible command-line workflow while retaining `notebooks/work_original.ipynb` unchanged for provenance.

The scripts use `random_state=42`. Numeric missing values are imputed using the training-set median before model fitting.

## Important data-sharing note

MIMIC datasets are credentialed-access resources. Do not upload raw MIMIC data, derived patient-level tables, or protected information to a public repository. Keep only code, documentation, and non-sensitive aggregate outputs in GitHub.
