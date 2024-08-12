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
from utils.data_processing import load_data, clean_data, preprocess_data, feature_engineering, monthly_summary_and_recommendations, handle_outliers
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

@app.route('/')
def index():
    try:
        # Train model
        model = train_model(featured_data)
        return render_template('index.html', data=featured_data.to_dict(orient='records'), model=model)
    except Exception as e:
        return f"Error in index route: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        future_data_input = request.form.get('future_data')
        future_data = pd.read_csv(StringIO(future_data_input))
        model = train_model(featured_data)
        predictions = predict_future_expenditure(model, future_data)
        return render_template('index.html', predictions=predictions.to_dict(orient='records'))
    except Exception as e:
        return f"Error in predict route: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=6002)
