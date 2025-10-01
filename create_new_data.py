import pandas as pd
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
data_folder = '../data'
new_data_file = os.path.join(data_folder, 'new_data.csv')
os.makedirs(data_folder, exist_ok=True)

# -----------------------------
# Number of synthetic samples
# -----------------------------
n_samples = 100  

# -----------------------------
# Feature names (same as train.csv minus 'Id' and 'Cover_Type')
# -----------------------------
features = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4'
] + [f'Soil_Type{i}' for i in range(1, 41)]

# -----------------------------
# Generate random synthetic data
# -----------------------------
np.random.seed(42)
data = pd.DataFrame({
    'Elevation': np.random.randint(1000, 4000, n_samples),
    'Aspect': np.random.randint(0, 360, n_samples),
    'Slope': np.random.randint(0, 60, n_samples),
    'Horizontal_Distance_To_Hydrology': np.random.randint(0, 5000, n_samples),
    'Vertical_Distance_To_Hydrology': np.random.randint(-500, 500, n_samples),
    'Horizontal_Distance_To_Roadways': np.random.randint(0, 5000, n_samples),
    'Hillshade_9am': np.random.randint(0, 255, n_samples),
    'Hillshade_Noon': np.random.randint(0, 255, n_samples),
    'Hillshade_3pm': np.random.randint(0, 255, n_samples),
    'Horizontal_Distance_To_Fire_Points': np.random.randint(0, 5000, n_samples),
    'Wilderness_Area1': np.random.randint(0,2,n_samples),
    'Wilderness_Area2': np.random.randint(0,2,n_samples),
    'Wilderness_Area3': np.random.randint(0,2,n_samples),
    'Wilderness_Area4': np.random.randint(0,2,n_samples),
})

# Add Soil_Type1 to Soil_Type40
for i in range(1, 41):
    data[f'Soil_Type{i}'] = np.random.randint(0,2,n_samples)

# -----------------------------
# Add Id column
# -----------------------------
data.insert(0, 'Id', range(1, n_samples + 1))

# -----------------------------
# Save CSV
# -----------------------------
data.to_csv(new_data_file, index=False)
print(f"Synthetic new data created at {new_data_file}")
