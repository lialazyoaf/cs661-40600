# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file is used preprocess the data.

import pandas as pd

# Load data from csv file
def load_data(file_path):
    '''Load data from csv file'''
    data = pd.read_csv(file_path)
    return data

# Clean data
def clean_data(data):
    '''Clean data by filling missing values and correcting data types'''
    # Fill missing values
    data.fillna({
        'Notes': '',
        'Amount': 0,
        'Transaction Type': '',
        'Category': '',
        'Account Name': '',
    }, inplace=True)
    # Convert 'Amount' to numeric
    data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')
    # Remove rows where 'Amount' could not be converted to numeric
    data.dropna(subset=['Amount'], inplace=True)
    
    return data

# Preprocess data
def preprocess_data(data):
    '''Preprocess data by formatting dates and extracting features'''
    # Convert 'Date' to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', format='%m/%d/%Y')
    # Extract year and month
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    # Drop rows where 'Date' could not be converted to datetime
    data.dropna(subset=['Date'], inplace=True)

    return data

# Feature engineering
def feature_engineering(data, budget_data):
    '''Perform feature engineering by associating categories with broad categories and merging with budget data.'''
    
    # Define broad categories and keywords for secondary categories
    broad_categories = {
        'Housing': ['Mortgage & Rent', 'Home Improvement', 'Utilities'],
        'Food & Dining': ['Groceries', 'Restaurants', 'Fast Food', 'Coffee Shops', 'Alcohol & Bars'],
        'Transportation': ['Gas & Fuel', 'Auto Insurance'],
        'Entertainment': ['Movies & DVDs', 'Music', 'Television', 'Electronics & Software'],
        'Utilities': ['Mobile Phone', 'Internet'],
        'Financial': ['Credit Card Payment', 'Paycheck'],
        'Personal Care': ['Haircut'],
        'Miscellaneous': ['Shopping', 'Miscellaneous']
    }

    # Function to assign broad category based on secondary category
    def assign_broad_category(category):
        for broad_cat, keywords in broad_categories.items():
            if category in keywords:
                return broad_cat
        return 'Uncategorized'  # If no match found

    # Apply the function to assign broad categories
    data['Broad_Category'] = data['Category'].apply(assign_broad_category)
    
    # Merge with budget data
    merged_data = pd.merge(data, budget_data, how='left', left_on='Category', right_on='Category')
    
    # Fill missing budget values with 0
    merged_data['Budget'].fillna(0, inplace=True)
    
    # Calculate the difference between actual spending and budgeted amount
    merged_data['Difference'] = merged_data['Amount'] - merged_data['Budget']
    
    return merged_data
