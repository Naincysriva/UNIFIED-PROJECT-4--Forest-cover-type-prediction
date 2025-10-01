import pandas as pd
import joblib
import os

# Paths
new_data_path = '../data/new_data.csv'
model_path = '../models/forest_cover_model.pkl'

# Load new data
df_new = pd.read_csv(new_data_path)
print("New data loaded!")

# Use Id column if exists
ids = df_new['Id'] if 'Id' in df_new.columns else pd.Series(range(1, len(df_new)+1))

# Prepare features (drop Id if exists)
X_new = df_new.drop(columns=['Id'], errors='ignore')

# Load trained model
model = joblib.load(model_path)
print("Model loaded!")

# Predict
y_pred = model.predict(X_new)

# Save predictions
results = pd.DataFrame({
    'Id': ids,
    'Predicted_Cover_Type': y_pred
})
results.to_csv('../data/new_data_predictions.csv', index=False)
print("Predictions saved to '../data/new_data_predictions.csv'")
print(results.head(10))
