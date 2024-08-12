# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file is used to define and train the model.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load and preprocess data function remains the same
def load_and_preprocess_data(expenditure_file_path, budget_file_path):
    expenditure_data = pd.read_csv(expenditure_file_path)
    budget_data = pd.read_csv(budget_file_path)
    
    # Clean and preprocess data
    cleaned_data = clean_data(expenditure_data)
    processed_data = preprocess_data(cleaned_data)
    featured_data = feature_engineering(processed_data, budget_data)
    
    return featured_data

# Hyperparameter tuning for Random Forest
def optimize_random_forest(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                              cv=3, n_jobs=-1, verbose=2, scoring='r2')

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and best score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best R^2 Score: {grid_search.best_score_}")

    return grid_search.best_estimator_

# Training the model
def train_model(data):
    # Create feature set and target variable
    X = data[['Day', 'Month', 'Year', 'Day_of_Week', 'Is_Weekend', 'Quarter', 'Cumulative_Spending']]
    X = pd.get_dummies(X)  # Convert categorical variables into dummy/indicator variables
    y = data['Adjusted_Amount']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimize Random Forest model
    model = optimize_random_forest(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    return model

# Example usage
if __name__ == "__main__":
    expenditure_file_path = 'data/expenditure.csv'
    budget_file_path = 'data/Budget.csv'
    data = load_and_preprocess_data(expenditure_file_path, budget_file_path)
    
    # Train the optimized model
    model = train_model(data)
