# visualize_predictions.py

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Ensure matplotlib works in Windows
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
predictions_path = '../data/new_data_predictions.csv'

# -----------------------------
# Load Predictions
# -----------------------------
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"Predictions file not found at {predictions_path}")

df_pred = pd.read_csv(predictions_path)
print("Predictions loaded!")

# -----------------------------
# Count of predicted cover types
# -----------------------------
cover_counts = df_pred['Predicted_Cover_Type'].value_counts().sort_index()

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,6))
cover_counts.plot(kind='bar', color='forestgreen')
plt.title("Predicted Forest Cover Types Distribution")
plt.xlabel("Cover Type")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
