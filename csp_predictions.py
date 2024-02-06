from datetime import timedelta
import pandas as pd
import numpy as np
import statistics
from CSP_generics import Variable, Constraint, CSP
from CSP_solver import Arc_Consistency
import pickle
import os
from ml import feature_eng, data_preparation, modelling
import matplotlib.pyplot as plt


def calculate_volatility(datasets):
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


def build_portfolio_csp(ds_names, min_investment, max_investment, risk_factor, asset_volatilities):
    if max_investment<=100:
        domain = np.arange(0, max_investment+10, 10)
    elif (max_investment>100 and  max_investment<=300):
        domain = np.arange(0, max_investment + 50, 50)
    else:
        domain = np.arange(0, max_investment + 10, 100)

    variables = []

    for name in ds_names:
        variables.append(Variable(name, domain))



    def calculate_portfolio_volatility(*values):

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

    return CSP("Portfolio Optimization", variables, constraints)


def csp_solver(csp, asset_volatilities):
    arc_solver = Arc_Consistency(csp)

    all_solutions = arc_solver.solve_all_wrapper()

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
    dataset_names = ["AMD", "CSCO", "QCOM", "SBUX", "TSLA"]
    datasets = [dataset + '.csv' for dataset in dataset_names]
    asset_volatilities = calculate_volatility(datasets)

    prediction_data = {}
    last_y_train_values = {}
    datasets = []

    for dataset_name in dataset_names:
        model = load_model(dataset_name)
        dataset = dataset_name+'.csv'


        if model is not None:
            print(f"Model loaded successfully for {dataset_name}")
        else:
            modelling(dataset)
            model = load_model(dataset_name)
            print(f"Model loaded successfully for {dataset_name}")

        datasets.append(dataset)
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


    csp = build_portfolio_csp(datasets, min_investment, max_investment, risk_factor, asset_volatilities)
    solutions = csp_solver(csp, asset_volatilities)
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

    final_solution = {}
    for x, y in best_solution.items():
        if y!=0:
            final_solution[x] = y
    assets = []
    allocations = []

    for x, y in final_solution.items():
        var_name = x.name.replace(".csv" , "")
        assets.append(var_name)
        allocations.append(y)

    os.makedirs('static', exist_ok=True)
    colors = ['mediumpurple','rebeccapurple','blueviolet','indigo','plum']

    plt.pie(allocations, colors=colors,labels=assets)
    plt.axis('equal')

    plt.savefig("static/piegraph.jpg", dpi=300)

    best_solution_str = {key: round(value, 2) for key, value in final_solution.items()}
    print(str(best_solution_str) + '. Expected return: ' + str(best_return))


    return final_solution
