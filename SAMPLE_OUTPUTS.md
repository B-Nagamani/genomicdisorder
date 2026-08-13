# Sample inputs and expected outputs

These examples are meant to show what the UI generally returns for a few typical cases. Numbers are samples — exact results depend on the trained models in the repo.

### Example A — typical case
Dashboard values:
- `Heart Rate (rates/min`: 120
- `Respiratory Rate (breaths/min)`: 30
- `White Blood cell count (thousand per microliter)`: 14.5
- `Blood test result`: abnormal
- `Genes in mother's side`: Yes
- `Inherited from father`: No
- `Maternal gene`: No
- `Paternal gene`: No
- `History of anomalies in previous pregnancies`: No
- `Assisted conception IVF/ART`: No

Sample output shown on `result.html`:
- Prediction: Multifactorial
- Confidence: ~87%
- Top subclass probabilities (example):
  - Mitochondrial — 45%
  - Single-gene — 25%
  - Chromosomal — 12%

> Note: those percentages are illustrative. Your actual numbers depend on the saved models and encoders.

### Example B — many missing / unknown inputs
If users leave fields blank or enter values the encoders never saw, `app.py` maps unseen categorical values to `-1` and numeric misses to `-1`.

Expected behavior:
- The server will attempt prediction and return a label with a confidence score (confidence may be low).
- If an unexpected error occurs during encoding, `result.html` will show the error message.

### Try it locally
1. Start the server from the project root:

```powershell
python app.py
```

2. Open `http://127.0.0.1:5001/`, register/login, then use the Dashboard to submit values.

3. For repeatable results, keep the joblib artifacts in place and run with the same package versions used during training.

### Implementation notes
- The app converts model predictions back to labels using `target_encoder_10.joblib` and `subclass_encoder_10.joblib`.
- If models are retrained with a different target label set, the mapping will change accordingly.
