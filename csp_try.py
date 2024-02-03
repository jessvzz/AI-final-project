from datetime import datetime, date, timedelta
import pandas as pd
import ta
import numpy as np
import statistics
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from CSP_generics import Variable, Constraint, CSP
from CSP_solver import Arc_Consistency
import pickle
import os

def data_preparation(dataset_name):
    df = pd.read_csv(dataset_name)

    """
    ------ Data preparation ------
    """

    # converting dates in yyyy-mm-dd format
    df['Date'] = pd.to_datetime(df['Date'])

    # renamed the column for a pure personal preference
    df.rename(columns={'Close/Last': 'Close'}, inplace=True)

    # converted to purely numeric values
    df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)
    df['Open'] = df['Open'].replace('[\$,]', '', regex=True).astype(float)
    df['High'] = df['High'].replace('[\$,]', '', regex=True).astype(float)
    df['Low'] = df['Low'].replace('[\$,]', '', regex=True).astype(float)

    df = df.ffill()
    return df

def calculate_volatility():
    datasets = ['AAPL.csv', 'AMZN.csv', 'SBUX.csv']
    volatilities = {}
    for dataset in datasets:
        df = data_preparation(dataset)
        # calculating relative price: proportional change in stock price between a day and the dau before
        df['Price relative'] = ""
        for i in range(1, len(df.Date)):
            df.loc[i, 'Price relative'] = df['Close'][i] / df['Close'][i - 1]

        # calculating proportional change in stock price between a day and the dau before
        df['Daily Return'] = ""
        for i in range(1, len(df.Date)):
            df.loc[i, 'Daily Return'] = np.log(df['Close'][i] / df['Close'][i - 1])

        # daily volatility
        DailyVolatility = statistics.stdev(df['Daily Return'][1:])
        #print("The daily volatility  is: {:.2%}".format(DailyVolatility))

        # annulized daily voltility
        AnnualizedDailyVolatilityCalendarDays = DailyVolatility * np.sqrt(365)
        #print("The annualized daily volatility measured in calendar days is: {:.2%}".format(
            #AnnualizedDailyVolatilityCalendarDays))
        AnnualizedDailyVolatilityCalendarDays = round(AnnualizedDailyVolatilityCalendarDays, 2)
        volatilities.update({dataset : AnnualizedDailyVolatilityCalendarDays})
    #print(volatilities)
    return volatilities

asset_volatilities = calculate_volatility()

def build_portfolio_csp(min_investment, max_investment, risk_factor, min_expected_return):
    domain = np.arange(0, max_investment+10, 10)
    max_for_each = max_investment / 3
    aapl = Variable('AAPL.csv', domain)
    amzn = Variable('AMZN.csv', domain)
    stbuks = Variable('SBUX.csv', domain)
    # [...]

    variables = [aapl, amzn, stbuks]  # ...

    def calculate_portfolio_volatility(*values):
        #asset_volatilities = calculate_volatility()

        #asset_volatilities =  {'AAPL.csv': 0.4, 'AMZN.csv': 0.4, 'SBUX.csv': 0.4}


        # Calcolare la somma delle volatilità ponderate
        weighted_volatilities_sum = sum(
            value * asset_volatilities[asset] for value, asset in zip(values, asset_volatilities))

        # Calcolare la somma totale degli investimenti
        total_investment = sum(values)
        if total_investment == 0:
            return 0


        portfolio_volatility = weighted_volatilities_sum / total_investment


        return portfolio_volatility
    """
    def calculate_min_return(*values):
        y_train = []
        y_test = []
        err=[]
        var_values = dict(zip(variables, values))
        filenames = [var_values[var] for var in variables]
        for file in filenames :
            df = pd.read_csv(file)
            df = df.dropna()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.dropna()

            #splitting the dataset
            X = df.drop('Close', axis=1)
            Y = df['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            last_value_test = y_test[-1]
            last_value_train = y_train[-1]

            err.append((last_value_test - last_value_train)*(values[filenames.index(file)]/last_value_test))
        err_value=sum(err)
        return err_value
    """

    constraints = []

    constraints.append(Constraint(scope=variables, condition=lambda *values: sum(values) <= max_investment))
    constraints.append(Constraint(scope=variables, condition=lambda *values: sum(values) >= min_investment))
    constraints.append(
        Constraint(scope=variables, condition=lambda *values: calculate_portfolio_volatility(*values) <= risk_factor))
    # constraints.append(Constraint(scope=variables, condition=lambda *values: calculate_min_return(*values) >= min_expected_return))
    # constraints.append(Constraint(scope=variables, condition=lambda *values: all(value >= 0 and value <= max_for_each for value in values)))
    return CSP("Portfolio Optimization", variables, constraints)


def csp_solver(csp):
    arc_solver = Arc_Consistency(csp)

    all_solutions = arc_solver.solve_all_wrapper()
    """"
    for solution in all_solutions:
        print("Solution:", solution)
    """
    return all_solutions


def load_model(dataset_name):
    models_directory = 'models'
    model_filename = f'model_{dataset_name}.pkl'
    model_path = os.path.join(models_directory, model_filename)

    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        print(f"Model file not found for {dataset_name}")
        return None

def main():
    dataset_names = ["AAPL", "AMZN"]

    for dataset_name in dataset_names:
        loaded_model = load_model(dataset_name)

        if loaded_model is not None:
            print(f"Model loaded successfully for {dataset_name}")
        else:
            print(f"Failed to load model for {dataset_name}")






"""
   csp = build_portfolio_csp(50, 60, 0.45, 0)
    solutions = csp_solver(csp)
    for solution in solutions:
        print("Solution: ", solution)
"""


main()
calculate_volatility()
"""
TODO'S
make domains and possibile values multiples of five
"""