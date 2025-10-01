# evaluate_model.py

import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Paths
# -----------------------------
data_path = '../data/train.csv'  # Agar test data alag nahi hai to train.csv bhi use kar sakte ho
model_path = '../models/forest_cover_model.pkl'

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(data_path)
X_test = df.drop(columns=['Id', 'Cover_Type'])
y_test = df['Cover_Type']

# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = joblib.load(model_path)
print("Model loaded successfully!")

# -----------------------------
# Predict on Test Data
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Confusion Matrix & Classification Report
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)
