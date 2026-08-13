import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

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

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv('train.csv')
    
    # Drop rows without the target variable
    df = df.dropna(subset=['Genetic Disorder'])
    
    X = df[FEATURE_COLS_IN_FILE].copy()
    y = df['Genetic Disorder']
    
    # Handle missing values
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna("Unknown").astype(str)
        else:
            X[col] = X[col].fillna(-1)
            
    # Label encode categorical variables
    encoders = {}
    categorical_dims = {}
    categorical_columns = []
    
    for i, col in enumerate(X.columns):
        if not pd.api.types.is_numeric_dtype(X[col]) or X[col].dtype == 'object' or str(X[col].dtype) in ['string', 'category']:
            le = LabelEncoder()
            # Convert everything to string first
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col]).astype(int)
            encoders[col] = le
            categorical_dims[col] = len(le.classes_)
            categorical_columns.append(i)
        
        # Ensure numerical columns are numeric
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)
            
    # Target encoding
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    joblib.dump(target_le, 'target_encoder.joblib')
    joblib.dump(encoders, 'feature_encoders.joblib')
    joblib.dump(list(X.columns), 'disorder_type_features.joblib') # To keep compatibility with UI
    
    return X, y_encoded, categorical_columns, list(categorical_dims.values()), target_le.classes_

def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n--- Training XGBoost (Machine Learning) ---")
    model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        use_label_encoder=False, 
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"XGBoost Train Accuracy: {train_acc:.4f}")
    print(f"XGBoost Test Accuracy: {test_acc:.4f}")
    
    joblib.dump(model, 'disorder_type_model_xgboost.joblib')
    print("Saved XGBoost model as 'disorder_type_model_xgboost.joblib'")
    return model

def train_tabnet(X_train, X_test, y_train, y_test, cat_idxs, cat_dims):
    print("\n--- Training TabNet (Deep Learning) ---")
    # TabNet requires numpy arrays
    X_train_np = X_train.values
    X_test_np = X_test.values
    
    # Prepare TabNet model
    clf = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax' # "sparsemax"
    )
    
    clf.fit(
        X_train=X_train_np, y_train=y_train,
        eval_set=[(X_train_np, y_train), (X_test_np, y_test)],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy'],
        max_epochs=50, patience=10,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Evaluate
    preds = clf.predict(X_test_np)
    test_acc = (preds == y_test).mean()
    print(f"TabNet Test Accuracy: {test_acc:.4f}")
    
    # Save the model
    saving_path_name = "./disorder_type_model_tabnet"
    saved_filepath = clf.save_model(saving_path_name)
    print(f"Saved TabNet model to '{saved_filepath}'")
    return clf

if __name__ == "__main__":
    X, y, cat_idxs, cat_dims, classes = load_and_preprocess_data()
    print(f"Target classes: {classes}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)
    tabnet_model = train_tabnet(X_train, X_test, y_train, y_test, cat_idxs, cat_dims)
    
    print("\nTraining Complete! Both ML (XGBoost) and DL (TabNet) models are ready to be integrated with the UI.")
