# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file is used to define and train the model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# defining a function to load and preprocess the data
def load_and_preproess_data(expenditure_file_path, budget_file_path):
    '''Load and preprocess data'''
    # Load data
    data = pd.read_csv(expenditure_file_path)
    budget_data = pd.read_csv(budget_file_path)
    # Clean and preprocess data
    cleaned_data = clean_data(data)
    processed_data = preprocess_data(cleaned_data)
    # Perform feature engineering
    feature_data = feature_engineering(processed_data, budget_data)
    
    return feature_data

# defining a function to train linear regression model
def train_model(data):
    # Define features and target
    X = data[['Month', 'Year']]
    y = data['Amount']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # loading and training the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # predicting and evaluating the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    return model

# defining a function to make predictions
def predict_future_expenditure(model, future_data):
    # Preprocess future data
    future_data = preprocess_data(future_data)
    future_data = clean_data(future_data)
    
    # Extract features
    X_future = future_data[['Month', 'Year']]
    
    # Make predictions
    predictions = model.predict(X_future)
    
    future_data['Predicted_Amount'] = predictions
    
    return future_data
