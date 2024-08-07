# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file contains the code for the Flask and Dash app.
# The Flask app serves the homepage and the Dash app serves the dashboard.

import pandas as pd
from utils.data_processing import load_data, clean_data, preprocess_data,feature_engineering
from flask import Flask, render_template, request
from models.model import load_and_preproess_data, train_model, predict_future_expenditure

app = Flask(__name__)
'''
# Load data
data = load_data('data/expenditure.csv')
budget_data = load_data('data/Budget.csv')
# Clean and preprocess data
cleaned_data = clean_data(data)
processed_data = preprocess_data(cleaned_data)
#perform feature engineering
feature_data = feature_engineering(processed_data, budget_data)
print(processed_data.head())
print(feature_data.head())
'''
@app.route('/')
def index():
    # Load data
    expenditure_data = load_data('data/expenditure.csv')
    budget_data = load_data('data/budget.csv')
    
    cleaned_data = clean_data(expenditure_data)
    preprocessed_data = preprocess_data(cleaned_data)
    featured_data = feature_engineering(preprocessed_data, budget_data)
    
    # Train model
    model = train_model(featured_data)
    
    # pass the model and featured data to the template
    return render_template('index.html', data=featured_data.to_dict(orient='records'), model=model)

@app.route('/predict', methods=['POST'])
def predict():
    # assume frontend will send the future data as a csv string
    future_data_input = request.form.get('future_data')
    
    # transform the csv string to a pandas dataframe
    future_data = pd.read_csv(pd.compat.StringIO(future_data_input))
    
    # Load model
    model = request.form.get('model')
    predictions = predict_future_expenditure(model, future_data)
    
    # return the predictions as a json object
    return render_template('index.html', predictions=predictions.to_dict(orient='records'))

if __name__ == '__main__':
    try:
        app.run(debug=True,port=6002)
    except Exception as e:
        print(f"Failed to start the server: {e}")

