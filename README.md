# deepfake-real-vs-fake-feature-classifier

This repository contains a full, reproducible pipeline for classifying `real` vs `fake` face images using handcrafted signal-processing features and classical ML models.

## What This Repo Contains

- Manifest builder with per-method stratified split:
  - `build_manifest.py`
- Feature extraction (intensity, residual, frequency, patch):
  - `extract_features.py`
- Model training and validation comparison:
  - `train_models.py`
- Final locked test evaluation:
  - `final_test_eval.py`

## What Is Not Included

Large local assets are intentionally excluded from Git:

- Raw datasets (`extracted/`)
- Augmented images (`styleclip_train_real_aug/`)
- Generated CSVs and artifacts (`train.csv`, `features/`, `analysis/`, `results/`)
- Local virtual environment (`.venv/`)

See `.gitignore` for full rules.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected Data Layout

Place datasets under `extracted/` in this form:

```text
extracted/
  CollabDiff/CollabDiff/{real,fake}/...
  stargan/stargan/{real,fake}/...
  starganv2/starganv2/{real,fake}/...
  styleclip/styleclip/{real,fake}/...
```

## Run Order

```bash
# 1) Build train/val/test manifests (+ styleclip train-real augmentation)
python build_manifest.py

# 2) Extract features
python extract_features.py

# 3) Analyze features
# (analysis script used in session; optional)

# 4) Train/compare models on val
python train_models.py

# 5) Final locked test evaluation (single run)
python final_test_eval.py
```

## Final Model (Session Result)

- Model: `GradientBoostingClassifier`
- Feature set: 10 non-patch features (frequency + residual + basic intensity)
- Test metrics:
  - F1 (fake): `0.8850`
  - ROC-AUC: `0.9458`
  - Precision: `0.8791`
  - Recall: `0.8911`

Per-method performance details are written to `results/final_results.json` when the pipeline is run locally.