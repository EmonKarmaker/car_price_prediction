from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load and clean dataset
car = pd.read_csv("Cleanned.csv")

# Ensure consistent data types and handle NaN for string columns
for col in ['company', 'name', 'fuel_type']:
    car[col] = car[col].fillna('Unknown').astype(str)

# Handle missing or invalid years before converting to int
car['year'] = car['year'].fillna(2000)  # Use default year if missing
car['year'] = car['year'].astype(float).astype(int)

# Handle missing or invalid kilometers
car['kms_driven'] = car['kms_driven'].fillna(0)
car['kms_driven'] = car['kms_driven'].astype(float).astype(int)

# Routes
@app.route('/', methods=['GET'])
def index():
    try:
        companies = sorted(car['company'].unique())
        car_models = sorted(car['name'].unique())
        years = sorted(car['year'].unique(), reverse=True)
        fuel_types = sorted(car['fuel_type'].unique())

        companies.insert(0, 'Select Company')
        car_models.insert(0, 'Select Model')
        years.insert(0, 'Select Year')
        fuel_types.insert(0, 'Select Fuel Type')

        return render_template(
            'index.html',
            companies=companies,
            car_models=car_models,
            years=years,
            fuel_types=fuel_types
        )
    except Exception as e:
        return f"Error loading page: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        # Validate and convert inputs
        if not year or year == 'Select Year':
            return "Error: Invalid Year"
        if not driven:
            return "Error: Kilometers driven required"

        year = int(float(year))
        driven = int(float(driven))

        # Create input dataframe for prediction
        input_data = pd.DataFrame(
            [[car_model, company, year, driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        # Predict
        prediction = model.predict(input_data)
        return str(np.round(prediction[0], 2))
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
