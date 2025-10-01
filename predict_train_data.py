import pandas as pd
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
data_path = '../data/train.csv'  # Train dataset
model_path = '../models/forest_cover_model.pkl'

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(data_path)
X = df.drop(columns=['Id', 'Cover_Type'])
y_true = df['Cover_Type']

# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)
print("Model loaded successfully!")

# -----------------------------
# Make Predictions
# -----------------------------
y_pred = model.predict(X)

# -----------------------------
# Compare predictions
# -----------------------------
results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
print(results.head(10))

# Optional: Accuracy check on train data
accuracy = (y_true == y_pred).mean()
print(f"Accuracy on training data: {accuracy:.4f}")
