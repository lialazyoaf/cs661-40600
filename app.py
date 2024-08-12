from flask import Flask, flash, render_template, redirect, session, url_for, request
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from firebase_config import auth
import plotly.graph_objects as go
import joblib

app = Flask(__name__)


#Initialize Firebase Authentication
app.secret_key = 'AIzaSyBbdk5jF2hKAo0zcPuec4GwQsPL0MBYkIg'

external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'
]

# Initialize Dash
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/', external_stylesheets=external_stylesheets)
dash_app.title = 'unique_dash_app'
app.layout = html.Div(id='dash-container')

#Load our model from model.py
model = joblib.load('model.pkl')

#Load datasets into pandas dataframe(df)
transactions_df = pd.read_csv(r'C:\Users\Hemanth\Downloads\BudgetBuddyUpdated\personal_transactions.csv')
budget_df = pd.read_csv(r'C:\Users\Hemanth\Downloads\BudgetBuddyUpdated\Budget.csv')

#Create a bar chart for spending by category
spending_by_category = transactions_df.groupby('Category')['Amount'].sum().reset_index()
fig = px.bar(spending_by_category, x='Category', y='Amount', title='Spending by Category')

#Create a pie chart for category and account type
fig_pie = px.pie(transactions_df, names='Category', values='Amount', title='Spending by Category and Account Type',
                 color='Account Name', hole=0.6)

#Line chart for amount over time
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
fig_line = px.line(transactions_df, x='Date', y='Amount', title='Amount Over Time')
fig_line.update_yaxes(tick0=0, dtick=5000)

#Create a bar chart for transactions by category
category_totals = transactions_df.groupby('Category')['Amount'].sum().reset_index()
category_totals = category_totals.sort_values(by='Amount', ascending=False)
fig_bar = px.bar(category_totals, x='Category', y='Amount', title='Transactions by Category', labels={'Amount': 'Total Amount'})
fig_bar.update_yaxes(tick0=0, dtick=5000)

#Create a bar chart for Description vs Amount
fig_desc_amount = px.bar(transactions_df, x='Description', y='Amount', title='Description vs Amount')
#Update y-axis to use a scale of 5,000
fig_desc_amount.update_yaxes(tick0=0, dtick=5000)


#Filter the DataFrame to include only Credit Card Payments
credit_card_payments_df = transactions_df[transactions_df['Category'] == 'Credit Card Payment']

#Calculate the total debt
total_debt = credit_card_payments_df['Amount'].sum()

#Create a pie chart to visualize the Credit Card Payments
fig_credit_card_payments = px.pie(credit_card_payments_df, names='Description', values='Amount', title='Credit Card Payments Distribution')


#Filter the data for Platinum Card and Silver Card transactions
filtered_df = transactions_df[transactions_df['Account Name'].isin(['Platinum card', 'Silver card'])]

#Agregate the total expenses for Platinum and Silver Card transactions

total_expenses_platinum = filtered_df[filtered_df['Account Name'] == 'Platinum card']['Amount'].sum()
total_expenses_silver = filtered_df[filtered_df['Account Name'] == 'Silver card']['Amount'].sum()

pie_data = pd.DataFrame({
    'Account Name': ['Platinum Card', 'Silver Card'],
    'Amount': [total_expenses_platinum, total_expenses_silver]
})

#Create a pie chart to visualize the Platinum and Silver Card transactions
fig_platinum_silver_card = px.pie(pie_data, names='Account Name', values='Amount', title='Platinum and Silver Card Transactions Distribution')


#Convert 'Date' column to datetime
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])

#Extract month and year from our 'Date' column
transactions_df['Month'] = transactions_df['Date'].dt.to_period('M').astype(str)

#we roup spending by month and calculate total amount
monthly_summary = transactions_df.groupby('Month')['Amount'].sum().reset_index()

#Create a bar chart for monthly expenses
fig_monthly_summary = px.bar(monthly_summary, x='Month', y='Amount', title='Monthly Summary of Expenses')

#we group the data by Account Name and calculate the total transactions
total_transactions = transactions_df.groupby('Account Name')['Amount'].sum().reset_index()

 #Populate a bar chart for the total transactions by card type
fig_total_transactions = px.bar(total_transactions, x='Account Name', y='Amount', title='Total Transactions by Card Type')

#we aggregate the data to calculate spending on different products and services
spending_summary = transactions_df.groupby('Description')['Amount'].sum().reset_index()

#Spending bar chart
fig_spending = px.bar(spending_summary, x='Description', y='Amount', title='Spending on Different Products and Services')
fig_spending.update_yaxes(tick0=0, dtick=5000)


#Aggregate the data to calculate Categories with most expenditure
category_spending = transactions_df.groupby('Category')['Amount'].sum().reset_index()

#we create a dictionary for easy lookup
spending_dict = category_spending.set_index('Category')['Amount'].to_dict()


#Creating a progress sphere to visualize the budget against monthly spending
#Calculate the total monthly budget
total_monthly_budget = budget_df['Budget'].sum()
print(f"Total Monthly Budget: ${total_monthly_budget:,.2f}")

#Calculate the total spending per month 
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
transactions_df['Month'] = transactions_df['Date'].dt.to_period('M')
monthly_spending = transactions_df.groupby('Month')['Amount'].sum().mean()


# Create a progress sphere
fig_progress = go.Figure(go.Indicator(
    mode="gauge+number",
    value=monthly_spending,
    title={'text': "Average Monthly Spending"},
    gauge={
        'axis': {'range': [0, total_monthly_budget]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, total_monthly_budget * 0.5], 'color': "lightgray"},
            {'range': [total_monthly_budget * 0.5, total_monthly_budget], 'color': "gray"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': total_monthly_budget
        }
    }
))

fig_progress.update_layout(title="Monthly Budget vs Spending")

# Dash layout
dash_app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Home', children=[
            html.H1('Welcome to Budget Buddy'),
            html.Div(className='card', children=[
                html.H2('Monthly Expenses Summary'),
                dcc.Graph(id='monthly-expenses-graph', figure=fig_monthly_summary),
                html.H3('Categories with the largest expense'),
                html.Div(className='horizontal-list', children=[
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-gas-pump'),
                        html.P('Gas & Fuel'),
                        html.P(f"${spending_dict.get('Gas & Fuel', 0):,.2f}")
                    ]),
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-tools'),
                        html.P('Home Improvement'),
                        html.P(f"${spending_dict.get('Home Improvement', 0):,.2f}")
                    ]),
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-shopping-cart'),
                        html.P('Groceries'),
                        html.P(f"${spending_dict.get('Groceries', 0):,.2f}")
                    ]),
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-home'),
                        html.P('Mortgage & Rent'),
                        html.P(f"${spending_dict.get('Mortgage & Rent', 0):,.2f}")
                    ]),
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-utensils'),
                        html.P('Restaurants'),
                        html.P(f"${spending_dict.get('Restaurants', 0):,.2f}")
                    ]),
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-wifi'),
                        html.P('Internet'),
                        html.P(f"${spending_dict.get('Internet', 0):,.2f}")
                    ]),
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-lightbulb'),
                        html.P('Utilities'),
                        html.P(f"${spending_dict.get('Utilities', 0):,.2f}")
                    ]),
                    html.Div(className='horizontal-list-item', children=[
                        html.I(className='fas fa-shopping-bag'),
                        html.P('Shopping'),
                        html.P(f"${spending_dict.get('Shopping', 0):,.2f}")
                    ])
                ])
            ])
        ]),
        dcc.Tab(label='Budget', value='tab-2'),
        dcc.Tab(label='Expense', value='tab-3'),
        dcc.Tab(label='Transaction', value='tab-4'),
        dcc.Tab(label='Debts', value='tab-5'),
        dcc.Tab(label='Credit Cards', value='tab-6'),
        dcc.Tab(label='Summary', value='tab-7')
    ]),
    html.Div(id='tabs-content')
])


# Callback to update tab content
@dash_app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Additional Analytics'),

            dcc.Graph(figure=fig_monthly_summary),
            dcc.Graph(figure=fig_pie),
            dcc.Graph(figure=fig_spending)
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Budget'),
            dcc.Graph(
                figure={
                    'data': [
                        {'x': budget_df['Category'], 'y': budget_df['Budget'], 'type': 'bar', 'name': 'Budget'}
                    ],
                    'layout': {
                        'title': 'Budget Overview'
                    }
                }
            ),

            dcc.Graph(figure=fig_progress)
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Expenses'),
            dcc.Graph(figure=fig_line),
            dcc.Graph(figure=fig_desc_amount)
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Transactions'),
            dcc.Graph(figure=fig_bar)
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Debts'),
            dcc.Graph(figure=fig_credit_card_payments)
        ])
    elif tab == 'tab-6':
        return html.Div([
            html.H3('Credit Cards'),
            dcc.Graph(figure=fig_platinum_silver_card),
            dcc.Graph(figure=fig_total_transactions)
        ])
    elif tab == 'tab-7':
        return html.Div([
            html.H3('Summary'),
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig_total_transactions)
            
        ])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form_register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user_with_email_and_password(email, password)
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except:
            flash('Failed to create account. Try again.', 'danger')
    return render_template('form_register.html')

@app.route('/form_log_in', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = user['idToken']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('predict'))
        except:
            flash('Login failed. Check your credentials.', 'danger')
    return render_template('form_log_in.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        future_data = request.form['future_data']
        try:
            #Replace NaN with null in the JSON string
            future_data = future_data.replace('NaN', 'null')
            future_data_df = pd.read_json(future_data)
            predictions = model.predict(future_data_df)
            future_data_df['Predicted_Amount'] = predictions
            return render_template('predict.html', predictions=future_data_df.to_dict(orient='records'))
        except ValueError as e:
            flash(f'Error processing data: {e}', 'danger')
            return render_template('predict.html')
    return render_template('predict.html')


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form['email']
        try:
            auth.send_password_reset_email(email)
            flash('Password reset email sent!', 'success')
            return redirect(url_for('login'))
        except:
            flash('Failed to send password reset email. Check the email address.', 'danger')
    return render_template('reset_password.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)





