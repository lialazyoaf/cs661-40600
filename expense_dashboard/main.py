# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file contains the code for the Flask and Dash app.
# The Flask app serves the homepage and the Dash app serves the dashboard.

import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from flask import Flask, render_template, request
import pandas as pd
from io import StringIO
from utils.data_processing import load_data, clean_data, preprocess_data, feature_engineering, monthly_summary_and_recommendations
from models.model import load_and_preprocess_data, train_model, predict_future_expenditure

app = Flask(__name__)

# Load and preprocess data
data = load_and_preprocess_data('data/expenditure.csv', 'data/budget.csv')

# Perform feature engineering and get monthly summary
featured_data = feature_engineering(data, load_data('data/budget.csv'))
monthly_spending, recommendations = monthly_summary_and_recommendations(featured_data)

print(featured_data.head())
print(monthly_spending.head())
print(recommendations.head())

'''index route: loading and preprocessing the data, training the model, and passing the model and featured data to the template.'''
@app.route('/')
def index():
    try:
        # Train model
        model = train_model(featured_data)
        # Pass the model and featured data to the template
        return render_template('index.html', data=featured_data.to_dict(orient='records'), model=model)
    except Exception as e:
        return f"Error in index route: {e}"
'''predict route: receiving the future data from the frontend, predicting the future expenditure using the trained model, and returning the predictions as a JSON object.'''
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assume frontend will send the future data as a csv string
        future_data_input = request.form.get('future_data')  
        # Transform the csv string to a pandas dataframe
        future_data = pd.read_csv(StringIO(future_data_input))
        # Load model (deserialize the model if needed)
        model = train_model(featured_data)
        predictions = predict_future_expenditure(model, future_data)
        # Return the predictions as a json object
        return render_template('index.html', predictions=predictions.to_dict(orient='records'))
    except Exception as e:
        return f"Error in predict route: {e}"

if __name__ == '__main__':
    try:
        app.run(debug=True, port=6002)
    except Exception as e:
        print(f"Failed to start the server: {e}")
