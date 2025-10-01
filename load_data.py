# preprocess_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load Dataset
data_path = '../data/train.csv'  # Ensure train.csv is in the data folder
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
print(df.head())

# 2. Preprocessing
# Drop unnecessary columns (like 'Id')
X = df.drop(columns=['Id', 'Cover_Type'])
y = df['Cover_Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Model Training
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
print("Training started on full dataset...")
model.fit(X_train, y_train)
print("Training finished!")

# 4. Accuracy Check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on test set: {accuracy:.4f}")

# 5. Save Model
os.makedirs('../models', exist_ok=True)
model_path = '../models/forest_cover_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to '{model_path}'")
