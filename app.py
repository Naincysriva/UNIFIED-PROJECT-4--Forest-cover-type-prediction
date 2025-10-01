# app.py
import streamlit as st
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Suppress Warnings
# -----------------------------
warnings.filterwarnings("ignore")

# -----------------------------
# Paths
# -----------------------------
model_path = '../models/forest_cover_model.pkl'

# -----------------------------
# Load Model
# -----------------------------
@st.cache_data
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Forest Cover Type Predictor", layout="wide")
st.title("🌲 Forest Cover Type Prediction")
st.markdown("Predict forest cover type based on terrain and soil features!")

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input_features():
    # Terrain features
    terrain = {
        'Elevation': st.sidebar.slider('Elevation', 1000, 4000, 2500),
        'Aspect': st.sidebar.slider('Aspect', 0, 360, 180),
        'Slope': st.sidebar.slider('Slope', 0, 60, 30),
        'Horizontal_Distance_To_Hydrology': st.sidebar.slider('Horizontal Distance To Hydrology', 0, 5000, 2000),
        'Vertical_Distance_To_Hydrology': st.sidebar.slider('Vertical Distance To Hydrology', -500, 500, 0),
        'Horizontal_Distance_To_Roadways': st.sidebar.slider('Horizontal Distance To Roadways', 0, 5000, 1000),
        'Hillshade_9am': st.sidebar.slider('Hillshade 9am', 0, 255, 100),
        'Hillshade_Noon': st.sidebar.slider('Hillshade Noon', 0, 255, 150),
        'Hillshade_3pm': st.sidebar.slider('Hillshade 3pm', 0, 255, 100),
        'Horizontal_Distance_To_Fire_Points': st.sidebar.slider('Horizontal Distance To Fire Points', 0, 5000, 1500),
    }

    # Wilderness Areas (4)
    wilderness = {}
    for i in range(1,5):
        wilderness[f'Wilderness_Area{i}'] = st.sidebar.selectbox(f'Wilderness Area {i}', [0,1], key=f'wa{i}')

    # Soil types grouped 4-4
    soil_types = {}
    for group in range(0,40,4):
        for i in range(1,5):
            soil_num = group + i
            soil_types[f'Soil_Type{soil_num}'] = st.sidebar.selectbox(f'Soil Type {soil_num}', [0,1], key=f'soil{soil_num}')

    data = {}
    data.update(terrain)
    data.update(wilderness)
    data.update(soil_types)

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Predicted Cover Type")
st.success(f"🌳 Predicted Forest Cover Type: **{prediction[0]}**")

st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=[f'Type {i}' for i in range(1,8)]).T
proba_df.columns = ['Probability']
st.bar_chart(proba_df)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Top 10 Feature Importances")
importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
top_features = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_features.values, y=top_features.index, palette="Greens_r")
plt.xlabel("Importance")
plt.title("Top 10 Important Features")
st.pyplot(plt)
