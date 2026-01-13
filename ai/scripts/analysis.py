import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# 1. DATA GENERATION 
# ---------------------------------------------------------
# Seed for reproducibility
np.random.seed(42)

# Load initial frame
df = pd.read_csv("<YOUR_PROXY_SERVER_LOGS>.csv")

new_records_count = 200
target_block_ratio = 0.3
last_timestamp = df['timestamp_us'].iloc[-1]
timestamps = [last_timestamp + (i + 1) * 60000000 for i in range(new_records_count)]
new_rows = []

for ts in timestamps:
    is_block = np.random.random() < target_block_ratio
    if is_block:
        decision = 'BLOCK'; traffic_class = 1; confidence = np.random.uniform(0.65, 0.79)
        ai_us = int(np.random.normal(70000, 5000)); client_fd = np.random.choice([6, 7, 8])
    else:
        decision = 'ALLOW'; traffic_class = 0; confidence = np.random.uniform(0.94, 0.99)
        ai_us = int(np.random.normal(50000, 3000)); client_fd = 5
        
    new_rows.append({'timestamp_us': ts, 'client_fd': client_fd, 'feature_us': 0, 'ai_us': ai_us,
                     'policy_us': 0, 'total_us': ai_us + 5, 'decision': decision,
                     'traffic_class': traffic_class, 'confidence': round(confidence, 6)})

final_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# 2. VISUALIZATION (Separate Windows)
# ---------------------------------------------------------

# Helper: Calculate relative time
start_time = final_df['timestamp_us'].min()
final_df['relative_time_s'] = (final_df['timestamp_us'] - start_time) / 1_000_000
colors = {'ALLOW': 'green', 'BLOCK': 'red'}
markers = {'ALLOW': 'o', 'BLOCK': 'x'}

# --- Window 1: Confidence Score Over Time ---
plt.figure(figsize=(10, 6), num='Confidence Over Time') # 'num' names the window
for decision, group in final_df.groupby('decision'):
    plt.scatter(group['relative_time_s'], group['confidence'], 
                label=decision, c=colors[decision], marker=markers[decision], alpha=0.6)
plt.title('Confidence Score Over Time')
plt.xlabel('Time (seconds from start)')
plt.ylabel('Confidence Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- Window 2: AI Latency Distribution ---
plt.figure(figsize=(8, 6), num='Latency Distribution')
ai_us_allow = final_df[final_df['decision'] == 'ALLOW']['ai_us']
ai_us_block = final_df[final_df['decision'] == 'BLOCK']['ai_us']
plt.boxplot([ai_us_allow, ai_us_block], labels=['ALLOW', 'BLOCK'], patch_artist=True)
plt.title('Distribution of AI Latency (ai_us)')
plt.ylabel('Microseconds (us)')
plt.grid(True, linestyle='--', alpha=0.5)

# --- Window 3: Confidence vs Latency ---
plt.figure(figsize=(10, 6), num='Confidence vs Latency')
for decision, group in final_df.groupby('decision'):
    plt.scatter(group['ai_us'], group['confidence'], 
                label=decision, c=colors[decision], marker=markers[decision], alpha=0.6)
plt.title('Correlation: Confidence vs. Latency')
plt.xlabel('AI Latency (ai_us)')
plt.ylabel('Confidence Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- Window 4: Counts ---
plt.figure(figsize=(8, 6), num='Decision Counts')
counts = final_df['decision'].value_counts()
bars = plt.bar(counts.index, counts.values, color=[colors.get(x, 'blue') for x in counts.index])
plt.title('Total Request Counts')
plt.ylabel('Number of Requests')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom', fontweight='bold')

# Show all windows at once
plt.show()