import pandas as pd
import glob

# Load multiple CICIDS CSV files
files = glob.glob(
        "../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ) + glob.glob(
        "../data/Tuesday-WorkingHours.pcap_ISCX.csv"
    )

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Select required columns
df = df[[
    "Total Length of Fwd Packets",
    "Fwd Header Length",
    "Total Fwd Packets",
    "Flow IAT Mean",
    "Label"
]]

# Rename to proxy feature names
df.columns = [
    "payload_size",
    "header_size",
    "request_count",
    "inter_arrival_us",
    "label"
]

# Clean invalid values
df = df.dropna()

# Convert ms → us
df["inter_arrival_us"] = (df["inter_arrival_us"] * 1000).astype(int)

df = df[df["payload_size"] >= 0]
df = df[df["header_size"] >= 0]
df = df[df["request_count"] > 0]

# Label reduction
def map_label(l):
    l = l.lower()
    if l == "benign":
        return "BENIGN"
    elif "bot" in l:
        return "BOT"
    else:
        return "ATTACK"

df["label"] = df["label"].apply(map_label)

# Save final dataset
df.to_csv("../data/dataset.csv", index=False)

print("dataset.csv generated")
print(df["label"].value_counts())
