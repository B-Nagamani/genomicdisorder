import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS_IN_FILE = [
    'Patient Age', "Genes in mother's side", 'Inherited from father',
    'Maternal gene', 'Paternal gene', 'Blood cell count (mcL)',
    "Mother's age", "Father's age", 'Status',
    'Respiratory Rate (breaths/min)', 'Heart Rate (rates/min',
    'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5',
    'Parental consent', 'Follow-up', 'Gender', 'Birth asphyxia',
    'Autopsy shows birth defect (if applicable)', 'Place of birth',
    'Folic acid details (peri-conceptional)', 'H/O serious maternal illness',
    'H/O radiation exposure (x-ray)', 'H/O substance abuse',
    'Assisted conception IVF/ART',
    'History of anomalies in previous pregnancies',
    'No. of previous abortion', 'Birth defects',
    'White Blood cell count (thousand per microliter)',
    'Blood test result',
    'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5',
]

print("Loading data for subclass model...")
df = pd.read_csv('train.csv')
df = df.dropna(subset=['Disorder Subclass'])

X = df[FEATURE_COLS_IN_FILE].copy()
y = df['Disorder Subclass']

# Preprocess X (same as before)
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna("Unknown").astype(str)
    else:
        X[col] = X[col].fillna(-1)

for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]) or X[col].dtype == 'object' or str(X[col].dtype) in ['string', 'category']:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col]).astype(int)
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)

# Encode Target
subclass_le = LabelEncoder()
y_encoded = subclass_le.fit_transform(y)
joblib.dump(subclass_le, 'subclass_encoder.joblib')

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("\n--- Training XGBoost for Subclass ---")
model = xgb.XGBClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.1, 
    use_label_encoder=False, 
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Subclass XGBoost Train Accuracy: {train_acc:.4f}")
print(f"Subclass XGBoost Test Accuracy: {test_acc:.4f}")

joblib.dump(model, 'subclass_model_xgboost.joblib')
print("Saved subclass model!")
