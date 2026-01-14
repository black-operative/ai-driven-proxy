import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

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

# ---------------------------------------------------------
# 1. LOAD ALL MODEL LOGS
# ---------------------------------------------------------
files = {
    "Random Forest"       : "../logs/random_forest_logs.csv",
    "XGBoost"             : "../logs/xgboost_logs.csv",
    "Gradient Boosting"   : "../logs/gradient_boosting_logs.csv",
    "Logistic Regression" : "../logs/logistic_regression_logs.csv"
}

dfs = []

for model, path in files.items():
    df = pd.read_csv(path)
    df["model"] = model
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

# Enforce consistent model ordering
final_df["model"] = pd.Categorical(
    final_df["model"],
    categories=files.keys(),
    ordered=True
)

# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
start_time = final_df["timestamp_us"].min()
final_df["relative_time_s"] = (
    final_df
    .groupby("model")["timestamp_us"]
    .transform(lambda x: (x - x.min()) / 1_000_000)
)

markers = {
    "ALLOW": "o",
    "BLOCK": "x"
}

# ---------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------
OUTPUT_DIR = "../figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# FIGURE 1: Confidence Over Time (Model Comparison)
# ---------------------------------------------------------
for model, group in final_df.groupby("model"):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        group["relative_time_s"],
        group["confidence"],
        alpha=0.6,
        s=18
    )

    plt.title(f"Confidence Over Time — {model}")
    plt.xlabel("Time (seconds from start)")
    plt.ylabel("Confidence")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/confidence_over_time_{model.replace(' ', '_').lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------
# FIGURE 2: AI Latency Distribution
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

data = [
    final_df[final_df["model"] == model]["ai_us"]
    for model in files.keys()
]

plt.boxplot(
    data, 
    labels=files.keys(), 
    patch_artist=True, 
    flierprops=dict(
        marker='o',
        markersize=3,
        alpha=0.4
    )
)
plt.title("AI Inference Latency Distribution")
plt.ylabel("AI Latency (µs)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ai_latency_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 3: Confidence vs Latency (Decision-aware)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for (model, decision), group in final_df.groupby(["model", "decision"]):
    plt.scatter(
        group["ai_us"],
        group["confidence"],
        label=f"{model} – {decision}",
        alpha=0.6,
        marker=markers.get(decision, "o")
    )

plt.title("Confidence vs AI Latency")
plt.xlabel("AI Latency (µs)")
plt.ylabel("Confidence")
plt.legend(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confidence_vs_latency.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 4: Decision Counts per Model (FIXED)
# ---------------------------------------------------------
counts = (
    final_df
    .groupby(["model", "decision"])
    .size()
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(10, 6))
counts.plot(kind="bar", stacked=True, ax=ax)

ax.set_title("Decision Counts per Model")
ax.set_ylabel("Number of Requests")
ax.set_xlabel("Model")
ax.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/decision_counts_per_model.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✓ All figures saved in ./{OUTPUT_DIR}/")
