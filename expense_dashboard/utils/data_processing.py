# linlan cai
# lc03159p@pace.edu
# cs 661 - Python Programming(40600)
# DR. BRIAN HARLEY
# Description: This file is used preprocess the data.

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load data from CSV file
def load_data(file_path):
    '''Load data from CSV file'''
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

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
    # Extract day, month, and year from 'Date'
    data['Day'] = data['Date'].dt.day
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    # Drop rows where 'Date' could not be converted to datetime
    data.dropna(subset=['Date'], inplace=True)
    
    return data

# Handle outliers
def handle_outliers(data):
    '''Identify and handle outliers in the data'''
    
    # Describe the data to identify potential outliers
    print("Summary statistics before handling outliers:")
    print(data['Amount'].describe())
    
    # Visualize initial data distribution with a box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data['Amount'], patch_artist=True, notch=True)
    plt.title('Box Plot of Amount Before Outlier Handling')
    plt.xlabel('Transactions')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.show()
    
    # Apply IQR method to detect and handle outliers
    Q1 = data['Amount'].quantile(0.25)
    Q3 = data['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping outliers at the upper bound
    data['Amount'] = data['Amount'].clip(lower_bound, upper_bound)
    
    # Redraw the box plot to confirm the outliers have been handled
    print("Summary statistics after handling outliers:")
    print(data['Amount'].describe())
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(data['Amount'], patch_artist=True, notch=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                flierprops=dict(marker='o', color='orange', markersize=5))
    
    plt.ylim([0, data['Amount'].max() * 1.1])
    plt.title('Box Plot of Amount After Outlier Handling')
    plt.xlabel('Transactions')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.show()

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
    income_categories = {
        'Income': ['Paycheck']
    }

    # Function to assign broad category based on secondary category
    def assign_broad_category(category):
        for broad_cat, keywords in {**broad_categories, **income_categories}.items():
            if category in keywords:
                return broad_cat
        return 'Uncategorized'  # If no match found
    
    # Apply the function to assign broad categories
    data['Broad_Category'] = data['Category'].apply(assign_broad_category)
    
    # Add new features based on date
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    data['Is_Weekend'] = data['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
    data['Quarter'] = data['Date'].dt.quarter
    
    # Adjust amount for income and expense
    data['Adjusted_Amount'] = data.apply(lambda row: row['Amount'] if row['Broad_Category'] == 'Income' else -row['Amount'], axis=1)
    
    # Historical feature: Cumulative spending up to the current date
    data['Cumulative_Spending'] = data.groupby(['Category'])['Adjusted_Amount'].cumsum()

    # Interaction features: Example of interaction between Month and Category
    data['Month_Category_Interaction'] = data['Month'].astype(str) + "_" + data['Category']

    # Merge with budget data, using suffixes to avoid column name conflicts
    merged_data = pd.merge(
        data, 
        budget_data, 
        how='left', 
        left_on='Category', 
        right_on='Category', 
        suffixes=('', '_budget')
    )
    print("Columns after merge:", merged_data.columns)
   
    # Fill missing budget values with 0
    merged_data['Budget'] = merged_data['Budget'].fillna(0)
    merged_data['Difference'] = merged_data['Amount'] - merged_data['Budget']
    merged_data['Balance'] = merged_data['Adjusted_Amount'].cumsum()

    return merged_data

# Monthly summary and recommendations
def monthly_summary_and_recommendations(data):
    '''Calculate monthly spending and generate recommendations'''
    # Group by Year, Month, and Category to get the monthly spending
    monthly_spending = data.groupby(['Year', 'Month', 'Category'])['Adjusted_Amount'].sum().reset_index()
    
    # Calculate recommendations based on spending and budget
    recommendations = []
    for _, row in monthly_spending.iterrows():
        category = row['Category']
        month_spent = row['Adjusted_Amount']
        budget = data.loc[data['Category'] == category, 'Budget_budget'].values[0]
        
        if month_spent > budget:
            recommendation = f"Reduce spending on {category} by {month_spent - budget:.2f} next month."
        else:
            recommendation = f"You are within the budget for {category}. Keep it up!"
        
        recommendations.append({
            'Year': row['Year'],
            'Month': row['Month'],
            'Category': category,
            'Spent': month_spent,
            'Budget': budget,
            'Recommendation': recommendation
        })
    recommendations_df = pd.DataFrame(recommendations)
    
    return monthly_spending, recommendations_df

    return monthly_spending, recommendations_df