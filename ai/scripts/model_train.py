import pandas as pd
import numpy as np
import pickle
import json
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

# -----------------------------
# Configurations
# -----------------------------
DATASET_PATH = "../data/dataset.csv"     # CHANGE THIS ACCORDINGLY
MODEL_DIR    = "../model/"               # CHANGE THIS ACCORDINGLY

FEATURES = [
    "payload_size",
    "header_size",
    "request_count",
    "inter_arrival_us"
]

LABEL_MAP = {
    "BENIGN": 0,
    "BOT": 1,
    "ATTACK": 2
}

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

X = df[FEATURES].values
y_raw = df["label"].map(LABEL_MAP).values

unique_y = np.unique(y_raw)
label_to_int = {val: i for i, val in enumerate(sorted(unique_y))}
y = np.array([label_to_int[val] for val in y_raw])

LABEL_NAMES = [name for name, code in LABEL_MAP.items() if code in unique_y]
num_classes = len(unique_y)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(f"{MODEL_DIR}scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# LSTM-ready labels (future extension)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# -----------------------------
# Train Models
# -----------------------------
models = {}
results = {}

print("Training Random Forest...")
models["random_forest"] = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15, 
    n_jobs=-1, 
    random_state=42
).fit(X_train, y_train)

print("Training XGBoost...")
models["xgboost"] = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
).fit(X_train_scaled, y_train)

print("Training Gradient Boosting...")
models["gradient_boosting"] = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
).fit(X_train_scaled, y_train)

print("Training Logistic Regression...")
models["logistic_regression"] = LogisticRegression(
    max_iter=1000, 
    n_jobs=-1, 
    random_state=42
).fit(X_train_scaled, y_train)

# -----------------------------
# Evaluate Models
# -----------------------------
print("\nMODEL EVALUATION")
for name, model in models.items():

    if name == "random_forest":
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:
        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    confidence = np.max(y_proba, axis=1)

    results[name] = {
        "accuracy": acc,
        "f1_score": f1,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confidence": confidence
    }

    print(f"\n{name.upper()}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

# -----------------------------
# Save Models
# -----------------------------
print("\nSaving models...")
for name, model in models.items():
    with open(f"{MODEL_DIR}{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# -----------------------------
# Save Metadata
# -----------------------------
metadata = {
    "features": FEATURES,
    "label_map": LABEL_MAP,
    "label_names": LABEL_NAMES,
    "num_classes": num_classes,
    "models": list(models.keys()),
    "results": {
        k: {
            "accuracy": float(v["accuracy"]),
            "f1_score": float(v["f1_score"]),
            "mean_confidence": float(v["confidence"].mean())
        }
        for k, v in results.items()
    }
}

with open(f"{MODEL_DIR}model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nTRAINING COMPLETE")