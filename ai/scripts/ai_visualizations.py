import matplotlib as mpl
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    }
)

MODEL_DIR = "../models/"   # CHANGE THIS ACCORDINGLY
OUTPUT_DIR = "../figures/" # CHANGE THIS ACCORDINGLY

# -----------------------------
# Load Metadata
# -----------------------------
with open(f"{MODEL_DIR}model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
LABEL_NAMES = metadata["label_names"]

# -----------------------------
# Load Models & Results
# -----------------------------
models = {}
results = {}

for name in metadata["models"]:
    with open(f"{MODEL_DIR}{name}_model.pkl", "rb") as f:
        models[name] = pickle.load(f)

# Load comparison CSV if needed
comparison_data = pd.DataFrame(
    {
        "Model"           : [k.replace("_", " ").title() for k in metadata["results"]],
        "Accuracy"        : [v["accuracy"] for v in metadata["results"].values()],
        "F1 Score"        : [v["f1_score"] for v in metadata["results"].values()],
        "Mean Confidence" : [v["mean_confidence"] for v in metadata["results"].values()],
    }
)

# -----------------------------
# Bar Charts
# -----------------------------
def bar_plot(x, y, title, ylabel, filename):
    plt.figure(figsize=(6, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}{filename}", dpi=300)
    plt.show()


bar_plot(
    comparison_data["Model"],
    comparison_data["Accuracy"],
    "Model Accuracy Comparison",
    "Accuracy",
    "model_accuracy_comparison.png",
)

bar_plot(
    comparison_data["Model"],
    comparison_data["F1 Score"],
    "Model F1 Score Comparison",
    "F1 Score",
    "model_f1_score_comparison.png",
)

bar_plot(
    comparison_data["Model"],
    comparison_data["Mean Confidence"],
    "Model Confidence Comparison",
    "Confidence",
    "model_confidence_comparison.png",
)

# -----------------------------
# Confusion Matrices
# -----------------------------
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
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
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}{filename}", dpi=300)
    plt.show()

# -----------------------------
# Feature Importance
# -----------------------------
tree_models = ["random_forest", "xgboost", "gradient_boosting"]

for name in tree_models:
    model = models[name]
    importance = model.feature_importances_

    df_imp = pd.DataFrame(
        {
            "Feature"    : FEATURES,
            "Importance" : importance
        }
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(6, 5))
    sns.barplot(data=df_imp, x="Importance", y="Feature")
    plt.title(f"{name.replace('_', ' ').title()} Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}feature_importance_{name}.png", dpi=300)
    plt.show()
