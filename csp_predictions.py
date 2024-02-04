from datetime import timedelta
import pandas as pd
import numpy as np
import statistics
from CSP_generics import Variable, Constraint, CSP
from CSP_solver import Arc_Consistency
import pickle
import os
from ml import feature_eng
from ml import data_preparation


def calculate_volatility():
    datasets = ['AAPL.csv', 'AMZN.csv']
    volatilities = {}
    for dataset in datasets:
        dataset_path = 'stocks/' + dataset
        df = data_preparation(dataset_path)
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

def build_portfolio_csp(min_investment, max_investment, risk_factor):
    domain = np.arange(0, max_investment+10, 10)
    max_for_each = max_investment / 3
    aapl = Variable('AAPL.csv', domain)
    amzn = Variable('AMZN.csv', domain)
    # [...]

    variables = [aapl, amzn]  # ...

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

def main(min_investment, max_investment, risk_factor):
    dataset_names = ["AAPL", "AMZN"]
    prediction_data = {}
    last_y_train_values = {}

    for dataset_name in dataset_names:
        model = load_model(dataset_name)

        if model is not None:
            print(f"Model loaded successfully for {dataset_name}")
        else:
            print(f"Failed to load model for {dataset_name}")
        dataset = dataset_name+'.csv'
        df = feature_eng(dataset)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna()
        df.set_index('Date', inplace=True)

        split_date = df.index.max() - timedelta(days=365)

        train = df[df.index <= split_date]
        test = df[df.index > split_date]

        # Separazione delle variabili indipendenti (X) dalle dipendenti (Y)
        X_train = train.drop('Close', axis=1)
        y_train = train['Close']
        X_test = test.drop('Close', axis=1)
        last_y_train_values[dataset] = y_train.iloc[-1]


        # Effettua le previsioni sul set di test (dati futuri)
        y_pred = model.predict(X_test)
        prediction = y_pred[-1]
        prediction_data[dataset] = prediction


    csp = build_portfolio_csp(min_investment, max_investment, risk_factor)
    solutions = csp_solver(csp)
    best_return = 0
    best_solution = None
    for solution in solutions:
        total_return = 0

        for variable, allocation in solution.items():
            dataset_name = variable.name
            prediction = prediction_data[dataset_name]
            last_asset_value = last_y_train_values[dataset_name]
            expected_return = prediction * (allocation / last_asset_value) - allocation

            total_return += expected_return
        if total_return > best_return:
            best_return = total_return
            best_solution = solution

    best_solution_str = {key: round(value, 2) for key, value in best_solution.items()}
    print(str(best_solution_str) + '. Expected return: ' + str(best_return))



"""
TODO'S
make domains and possibile values multiples of five
"""