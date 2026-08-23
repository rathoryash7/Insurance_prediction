from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib  # Used for loading the trained model

# Load pre-trained model and scaler
scaler = joblib.load('scaler.pkl')  # Ensure this matches the saved scaler file name
xgb_reg = joblib.load('xgb_model_optimized.pkl')  # Ensure this matches the saved model's name

# Load the column names used in one-hot encoding
columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('insurance-prediction.html')  # Render the prediction form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from HTML form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Prepare the input data
        input_data = {'age': age, 'sex_male': sex, 'bmi': bmi, 'children': children, 'smoker_yes': smoker}

        # Handle one-hot encoding for region
        region_mapping = {0: 'region_northwest', 1: 'region_southeast', 2: 'region_southwest'}
        input_data[region_mapping.get(region, '')] = 1  # Set the correct region to 1
        input_data = {k: v if k in input_data else 0 for k, v in zip(columns, [0] * len(columns))}  # Ensure all columns exist in the input data

        input_data_as_numpy_array = np.asarray(list(input_data.values()))
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Standardize the input data using the same scaler used for training
        input_data_standardized = scaler.transform(input_data_reshaped)

        # Make the prediction
        prediction = xgb_reg.predict(input_data_standardized)

        # Return the result to the HTML page
        return render_template('insurance-prediction.html', prediction=f'The predicted insurance cost is USD {prediction[0]:.2f}')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
