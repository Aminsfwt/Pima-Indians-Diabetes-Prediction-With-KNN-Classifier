from KNN_model import *

def predict_diabetes(input_data):
    """
    Predicts diabetes risk based on input features.
    
    Parameters:
        input_data (list): [Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    Returns:
        str: "Diabetes" or "No Diabetes"
    """

    input_scaled = scaler.transform([input_data])
    prediction = knn_grid.predict(input_scaled)
    return "Diabetes" if prediction[0] == 1 else "No Diabetes"


        

if __name__ == "__main__":
    # Example input data
    sample_data = [6, 148, 72, 35, 22, 33.6, 0.627, 50]

    # Make prediction
    result = predict_diabetes(sample_data)
    print(f"Prediction: {result}")
