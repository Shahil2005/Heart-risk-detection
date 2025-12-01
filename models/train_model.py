import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Constants
DATA_PATH_MNT = '../data/heart.csv'
DATA_PATH_LOCAL = '../data/heart.csv'
MODEL_PATH = 'heart-risk-project\models\models\heart_model.joblib'

# Expected columns (based on common heart disease datasets)
EXPECTED_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
TARGET_NAMES = ['target', 'heart_disease', 'disease', 'condition']

def load_data():
    """Load dataset from available path."""
    if os.path.exists(DATA_PATH_LOCAL):
        print(f"Loading data from {DATA_PATH_LOCAL}...")
        return pd.read_csv(DATA_PATH_LOCAL)
    elif os.path.exists(DATA_PATH_MNT):
        print(f"Loading data from {DATA_PATH_MNT}...")
        return pd.read_csv(DATA_PATH_MNT)
    else:
        print(f"Error: Dataset not found at {DATA_PATH_LOCAL} or {DATA_PATH_MNT}")
        sys.exit(1)

def train():
    """Main training pipeline."""
    df = load_data()
    
    # 1. Print Header
    print("CSV Header:", df.columns.tolist())
    
    # 2. Identify Target
    target_col = None
    for col in df.columns:
        if col.lower() in TARGET_NAMES:
            target_col = col
            break
    
    if not target_col:
        print("Error: Could not identify target column.")
        print(f"Looked for one of: {TARGET_NAMES}")
        print(f"Found columns: {df.columns.tolist()}")
        print("Please rename your target column in the CSV.")
        sys.exit(1)
        
    print(f"Identified target column: '{target_col}'")
    
    # 3. Validate/Map Features
    # We will try to use the expected columns if they exist, or map if possible.
    # For this strict requirement, we'll check if all EXPECTED_COLUMNS are present.
    # If not, we'll try to proceed with whatever columns are present MINUS the target,
    # but strictly speaking the prompt asks to "try to map... if ambiguous return error".
    # Let's assume if we have a mismatch in expected columns we should warn or fail.
    # However, to be robust, let's use the intersection or fail if criticals are missing.
    # The prompt says: "If column names differ, detect them automatically and adapt code; but if automatic detection is ambiguous, return an error"
    
    # Let's try to match case-insensitively
    df.columns = [c.lower() for c in df.columns]
    
    missing_features = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"Warning: The following expected columns were not found: {missing_features}")
        print("Attempting to use remaining columns as features.")
    
    feature_cols = [c for c in df.columns if c != target_col]
    
    # 4. Preprocessing
    X = df[feature_cols]
    y = df[target_col]
    
    # Detect numeric vs categorical
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 5. Pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(max_iter=1000))])
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    # 7. Evaluation
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("Could not calculate ROC AUC (possibly only one class in test set).")
    
    # 8. Save Model
    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    model_data = {
        'model': clf,
        'columns': feature_cols,
        'target_name': target_col
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Saved feature columns: {feature_cols}")

if __name__ == "__main__":
    train()
