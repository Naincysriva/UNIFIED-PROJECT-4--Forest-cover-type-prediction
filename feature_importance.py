import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
model_path = '../models/forest_cover_model.pkl'

# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)
print("Model loaded successfully!")

# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
top_features = importance.sort_values(ascending=False).head(10)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))
top_features.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))
top_features.plot(kind='barh')
plt.gca().invert_yaxis()  # Top feature upar
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")

# Replace plt.show() with plt.savefig()
plt.savefig('../models/feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

