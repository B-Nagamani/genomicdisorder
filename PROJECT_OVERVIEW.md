# Project Overview - Gene AI Clinical Screening

## Summary

## Key Components
  - Loads pre-trained models and encoders (`*_10.joblib` artifacts).
  - Renders templates in `templates/` and exposes routes: `/`, `/login`, `/register`, `/dashboard`, `/predict`, `/logout`.
  - `train_10_features.py` - trains the 10-feature XGBoost models, saves: `disorder_type_model_xgboost_10.joblib`, `subclass_model_xgboost_10.joblib`, `feature_encoders_10.joblib`, `top_10_features.joblib`, `target_encoder_10.joblib`, `subclass_encoder_10.joblib`.
  - `train_subclass.py` - trains a fuller-feature subclass model and saves `subclass_model_xgboost.joblib` and `subclass_encoder.joblib`.
  - `prepare_and_train.py` - data preprocessing pipeline and training helpers (XGBoost & TabNet), saves `disorder_type_model_xgboost.joblib`, `feature_encoders.joblib`, and related artifacts when run.
  - `check.py` - inspects and prints encoder classes from `feature_encoders.joblib`.

## Data & Model Artifacts (present)
  - `disorder_type_model_xgboost_10.joblib`
  - `subclass_model_xgboost_10.joblib`
  - `feature_encoders_10.joblib`
  - `top_10_features.joblib`
  - `target_encoder_10.joblib`, `subclass_encoder_10.joblib`
  - Full-feature artifacts: `disorder_type_model_xgboost.joblib`, `feature_encoders.joblib`, `disorder_type_features.joblib`

## High-level Flow
1. Training scripts read `train.csv` → preprocess → fit models → save joblib artifacts.
2. `app.py` loads the 10-feature models & encoders at startup.
3. Clinician logs in via UI → fills dashboard form with the top 10 clinical features → submits prediction.
4. `app.py` encodes inputs, runs `xgb_model.predict` and `subclass_model.predict_proba`, decodes labels and returns results in `result.html`.

## How to Run (prediction server)
1. Ensure the required Python packages are installed (e.g., Flask, pandas, joblib, xgboost, scikit-learn). Create a `venv` and install packages.
2. From the project folder run:

```powershell
python app.py
```

3. Open `http://127.0.0.1:5001/` in a browser, register/login, go to Dashboard, enter feature values and submit.

## Files of Interest

## Notes & Assumptions

## Next
See the visual flowchart in `FLOWCHART.md` and example predictions in `SAMPLE_OUTPUTS.md` for representative inputs & outputs.
# Project overview

This repository contains a small Flask application for clinical genetic screening. The web UI accepts a short set of clinical features, encodes the inputs, and uses pre-trained XGBoost models to predict a likely genetic disorder and the most probable subclasses.

## At a glance
- Purpose: provide a lightweight interface to run disorder and subclass predictions from patient clinical data.
- Interface: the server is implemented in `app.py` and serves pages from the `templates/` folder.

## Main parts
- `app.py` — Flask app that loads the saved model files and encoders, renders the UI, and handles prediction requests.
- `templates/` — HTML pages used by the UI (`landing.html`, `login.html`, `register.html`, `dashboard.html`, `result.html`).
- Training scripts:
  - `train_10_features.py` — trains the focused 10-feature XGBoost models and writes the `_10.joblib` artifacts used by the app.
  - `train_subclass.py` — trains a fuller-feature subclass model.
  - `prepare_and_train.py` — end-to-end preprocessing and training helpers (includes an optional TabNet training routine).
- Utility: `check.py` prints encoder class lists from saved encoders.

## Artifacts included
- Datasets: `train.csv`, `test.csv`.
- Model/encoder files (examples present in the repo): `disorder_type_model_xgboost_10.joblib`, `subclass_model_xgboost_10.joblib`, `feature_encoders_10.joblib`, `top_10_features.joblib`, `target_encoder_10.joblib`, `subclass_encoder_10.joblib`. Full-feature artifacts are available too (see `prepare_and_train.py`).

## How it works (runtime flow)
1. The server (`app.py`) loads the top-10 feature encoders and models at startup.
2. A user registers/logs in and fills the dashboard form with clinical values.
3. On submit, the server encodes values (categoricals via saved LabelEncoders, numerics coerced to numbers) and runs the disorder and subclass models.
4. Results are converted back to human labels and shown in `result.html` with a confidence score and top subclass probabilities.

## Run locally (quick)
1. Create a virtual environment and install the usual packages (`Flask`, `pandas`, `joblib`, `scikit-learn`, `xgboost`).

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt  # or install packages manually
python app.py
```

2. Open `http://127.0.0.1:5001/` in your browser, register, and use the Dashboard to try predictions.

## Files to look at first
- [app.py](app.py)
- [train_10_features.py](train_10_features.py)
- [train_subclass.py](train_subclass.py)
- [prepare_and_train.py](prepare_and_train.py)
- [check.py](check.py)
- Templates: [templates/landing.html](templates/landing.html), [templates/dashboard.html](templates/dashboard.html), [templates/result.html](templates/result.html)

## Notes
- No code changes were made — the edits in this repo are documentation only.
- The app uses the `_10` joblib files by default. If you retrain and overwrite artifact names, update `app.py` accordingly.

If you want, I can also:
- Render the Mermaid flowchart to an image and add it to the repo, or
- Start the server here and run a single prediction using one of the saved models.
