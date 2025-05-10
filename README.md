# Pima Indians Diabetes Prediction

This repository contains a machine learning project that predicts diabetes risk using the Pima Indians Diabetes dataset.
The project preprocesses the data, scales the input features, and uses a tuned K-Nearest Neighbors (KNN) model to classify patients as having "Diabetes" or "No Diabetes".

## Repository Structure

```
Pima Indians Diabetes/
    ├── KNN_model.py         # Contains the definition, training, and hyperparameter tuning for the KNN model and scaler.
    ├── main.py              # Main script for making predictions from new input data.
    └── (Additional files and data as required)
```

## Overview

- **Data Preprocessing:**  
  The dataset was standardized using `StandardScaler` on features including:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age

- **Prediction Function:**  
  The `predict_diabetes` function (implemented in `main.py`) takes a list of input data, scales it with the same `StandardScaler` used during training, and then uses the tuned KNN model (`knn_grid`) to predict the diabetes risk. The function returns:
  - `"Diabetes"` if the model predicts a positive case.
  - `"No Diabetes"` otherwise.

- **Example Usage:**  
  Running `main.py` will use sample input data (e.g., `[6, 148, 72, 35, 0, 33.6, 0.627, 50]`) to demonstrate how predictions are made.

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn

Install the packages using:

````bash
pip install pandas numpy scikit-learn 
