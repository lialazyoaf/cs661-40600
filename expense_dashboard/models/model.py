# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file is used to define and train the model.
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from utils.data_processing import load_data, clean_data, preprocess_data, feature_engineering, handle_outliers

# Load and preprocess data function remains the same
def load_and_preprocess_data(expenditure_file_path, budget_file_path):
    expenditure_data = load_data(expenditure_file_path)
    budget_data = load_data(budget_file_path)
    
    if expenditure_data is None or budget_data is None:
        raise ValueError("Failed to load data. Please check the file paths and ensure the CSV files exist.")

    # Clean and preprocess data
    cleaned_data = clean_data(expenditure_data)
    processed_data = preprocess_data(cleaned_data)
    
    if processed_data is None:
        raise ValueError("Data preprocessing failed.")
    
    # Handle outliers
    processed_data = handle_outliers(processed_data)
    
    if processed_data is None:
        raise ValueError("Outlier handling failed.")

    # Feature engineering
    featured_data = feature_engineering(processed_data, budget_data)
    
    if featured_data is None:
        raise ValueError("Feature engineering failed.")
    
    return featured_data

# Training the Linear Regression model
def train_model(data):
    # Create feature set and target variable
    X = data[['Day', 'Month', 'Year', 'Day_of_Week', 'Is_Weekend', 'Quarter', 'Cumulative_Spending']]
    X = pd.get_dummies(X)  # Convert categorical variables into dummy/indicator variables
    y = data['Adjusted_Amount']
    
    # Standardize the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ridge regression with cross-validation
    alphas = [0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_scaled, y)
    
    # Predict and evaluate the model
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    return model

# Predicting future expenditures
def predict_future_expenditure(model, future_data):
    # Preprocess future data (similar steps to what was done during training)
    future_data = clean_data(future_data)
    future_data = preprocess_data(future_data)

    # Feature engineering for future data
    future_data = feature_engineering(future_data, pd.DataFrame(columns=['Category', 'Budget']))

    # Create feature set
    X_future = future_data[['Day', 'Month', 'Year', 'Day_of_Week', 'Is_Weekend', 'Quarter', 'Cumulative_Spending']]
    X_future = pd.get_dummies(X_future)  # Convert categorical variables into dummy/indicator variables

    # Make predictions
    predictions = model.predict(X_future)
    future_data['Predicted_Amount'] = predictions

    return future_data[['Date', 'Category', 'Predicted_Amount']]
