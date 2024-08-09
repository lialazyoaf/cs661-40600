# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file is used to define and train the model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_processing import preprocess_data, clean_data, feature_engineering

def load_and_preprocess_data(expenditure_file_path, budget_file_path):
    expenditure_data = pd.read_csv('data/expenditure.csv')
    budget_data = pd.read_csv('data/budget.csv')
    
    # Clean and preprocess data
    cleaned_data = clean_data(expenditure_data)
    processed_data = preprocess_data(cleaned_data)
    featured_data = feature_engineering(processed_data, budget_data)
    
    return featured_data

def train_model(data):
    # Create feature set and target variable
    X = data[['Day', 'Month', 'Year', 'Category']]
    X = pd.get_dummies(X)  # Convert categorical variables into dummy/indicator variables
    y = data['Adjusted_Amount']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    return model

def predict_future_expenditure(model, future_data):
    # Preprocess future data
    future_data = clean_data(future_data)
    future_data = preprocess_data(future_data)
    
    # Create feature set
    X_future = future_data[['Day', 'Month', 'Year', 'Category']]
    X_future = pd.get_dummies(X_future)  # Convert categorical variables into dummy/indicator variables
    
    # Ensure the future data has the same columns as the training data
    missing_cols = set(model.feature_importances_) - set(X_future.columns)
    for c in missing_cols:
        X_future[c] = 0
    X_future = X_future.reindex(columns=model.feature_importances_, fill_value=0)
    
    # Make predictions
    predictions = model.predict(X_future)
    
    future_data['Predicted_Amount'] = predictions
    
    return future_data

# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    expenditure_file_path = 'data/expenditure.csv'
    budget_file_path = 'data/budget.csv'
    data = load_and_preprocess_data(expenditure_file_path, budget_file_path)
    
    # Train the model
    model = train_model(data)
    
    # Example future data for prediction
    future_data = pd.DataFrame({
        'Date': ['01/01/2024', '02/01/2024'],
        'Description': ['', ''],
        'Amount': [0, 0],
        'Transaction Type': ['', ''],
        'Category': ['Groceries', 'Rent'],
        'Account Name': ['', '']
    })
    
    # Predict future expenditure
    future_predictions = predict_future_expenditure(model, future_data)
    print(future_predictions)

