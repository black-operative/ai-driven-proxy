import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "../data/dataset.csv"
MODEL_OUT = "../model/model.pkl"

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

LABEL_NAMES = list(LABEL_MAP.keys())

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)

X = df[FEATURES].values
y = df["label"].map(LABEL_MAP).values

# Get actual label names present in data
unique_y = np.unique(y)
LABEL_NAMES = [name for name, code in LABEL_MAP.items() if code in unique_y]

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
confidence = np.max(y_proba, axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# Feature Importance
# -----------------------------
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance_df)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=importance_df,
    x="importance",
    y="feature"
)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# -----------------------------
# Feature Correlation Heatmap
# -----------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    df[FEATURES].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# Confidence Analysis
# -----------------------------
confidence_df = pd.DataFrame({
    "true_label": y_test,
    "pred_label": y_pred,
    "confidence": confidence
})

plt.figure(figsize=(7, 4))
sns.histplot(
    data=confidence_df,
    x="confidence",
    hue="pred_label",
    bins=20,
    kde=True
)
plt.title("Prediction Confidence Distribution")
plt.tight_layout()
plt.show()

# Per-class confidence stats
print("\n=== Confidence Statistics by Predicted Class ===")
for name, cls in LABEL_MAP.items():
    subset = confidence_df[confidence_df["pred_label"] == cls]
    if not subset.empty:
        print(f"\n{name}")
        print(subset["confidence"].describe())

# -----------------------------
# Save artifacts
# -----------------------------
importance_df.to_csv("feature_importance.csv", index=False)
confidence_df.to_csv("prediction_confidence.csv", index=False)

with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to:", MODEL_OUT)
print("Artifacts saved:")
print(" - feature_importance.csv")
print(" - prediction_confidence.csv")
