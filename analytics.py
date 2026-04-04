import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#this is for generating anylytics for the experiments performed and stored in results folder
#note metrics in this use sklearn libraries while the ones in CSV files are calculated so there might be a little difference

prefix=input("Enter the files prefix name: ")
folder = "results"              

pattern = os.path.join(folder, f"{prefix}*.csv")
files = glob.glob(pattern)

print(f"Found {len(files)} files.")

# Load all into a list
runs = []

for file in sorted(files):
    df = pd.read_csv(file)
    df["source_file"] = os.path.basename(file)  # track which run it came from
    runs.append(df)

all_actual = []
all_predictions = []

for file in files:
    df = pd.read_csv(file)

    # Remove non-numeric rows
    df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()]
    df = df[pd.to_numeric(df.iloc[:, 1], errors='coerce').notnull()]

    predictions = df.iloc[:, 0].astype(float)
    actual = df.iloc[:, 1].astype(float)

    all_predictions.extend(predictions)
    all_actual.extend(actual)

plt.figure(figsize=(8, 8))

# Density plot
plt.hexbin(all_actual, all_predictions, gridsize=30, cmap='inferno')

# Color bar 
plt.colorbar(label='Point Density')

# Perfect prediction line
min_val = min(all_actual)
max_val = max(all_actual)
plt.plot([min_val, max_val], [min_val, max_val], color='cyan', linewidth=2)

# Labels
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Density Plot: Predicted vs Actual")

plt.grid(True)
plt.tight_layout()
plt.show()


r2_list = []
mse_list = []

for file in files:
    df = pd.read_csv(file)

    # Keep only numeric rows (ignore bottom metrics)
    df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()]
    df = df[pd.to_numeric(df.iloc[:, 1], errors='coerce').notnull()]

    predictions = df.iloc[:, 0].astype(float)
    actual = df.iloc[:, 1].astype(float)

    r2 = r2_score(actual, predictions)
    mse = mean_squared_error(actual, predictions)

    r2_list.append(r2)
    mse_list.append(mse)

# Compute summary stats
summary = {
    "Runs": len(files),
    "R2 Avg": np.mean(r2_list),
    "R2 Min": np.min(r2_list),
    "R2 Max": np.max(r2_list),
    "MSE Avg": np.mean(mse_list),
    "MSE Min": np.min(mse_list),
    "MSE Max": np.max(mse_list)
}

# Print summary
print("=== Experiment Summary ===")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")