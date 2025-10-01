# UNIFIED-PROJECT-4--Forest-cover-type-prediction

# Forest Cover Type Prediction

## Overview

This project predicts forest cover type from cartographic variables such as elevation, soil type, and wilderness area using machine learning.

## Dataset

The dataset consists of 15,120 samples from US Forest Service and US Geological Survey data with 54 features describing forest patches. The target is a categorical label representing forest cover type (7 classes).

## Features

- Quantitative features representing soil types, elevation, hydrologic variables, etc.
- Categorical features encoded as binary variables.

## Approach

- Data loading and cleaning.
- Feature engineering.
- Model training using Random Forest, XGBoost, and LightGBM.
- Hyperparameter tuning and evaluation.
- Visualization of feature importance and model metrics.

## Usage

1. Clone the repository:

```
git clone https://github.com/Naincysriva/UNIFIED-PROJECT-4--Forest-cover-type-prediction.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the training script or Jupyter notebook as provided to start training and evaluation.

## Results

- Provides classification accuracy and other metrics.
- Provides insights with feature importance plots.
