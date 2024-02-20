from datetime import timedelta
import pandas as pd
import numpy as np
import statistics
import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import warnings
from ml import feature_eng, data_preparation, modelling
from CSP_generics import Variable, Constraint, CSP
from CSP_solver import Arc_Consistency

def calculate_volatility(datasets):
    """
    :return: a dictionary, where the keys are the datasets and the values are the volatilities
    """
    volatilities = {}

    for dataset in datasets:
        # constructing full path to the dataset
        dataset_path = os.path.join('stocks', dataset)

        # loading and prepare the dataset
        df = data_preparation(dataset_path)

        # calculating relative price: proportional change in stock price between a day and the day before
        df['Price relative'] = df['Close'].pct_change() + 1

        # calculating proportional change in stock price between a day and the day before
        df['Daily Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # calculate daily volatility
        daily_volatility = statistics.stdev(df['Daily Return'].dropna())

        # annualize daily volatility using 365 days
        annualized_daily_volatility_calendar_days = daily_volatility * np.sqrt(365)
        annualized_daily_volatility_calendar_days = round(annualized_daily_volatility_calendar_days, 2)

        # storing the result in the dictionary
        volatilities[dataset] = annualized_daily_volatility_calendar_days

    return volatilities


def build_portfolio_csp(ds_names, min_investment, max_investment, risk_factor, asset_volatilities):
    """
    :return: A CSP for the portfolio optimization
    """
    # handling case where min investment is set to be higher than maximum investment
    if min_investment>max_investment:
        min_investment = 0

    # the increment in the domain changes as the max investment changes
    # done to reduce waiting times during CSP
    if max_investment<=100:
        incr =10
    elif max_investment>100 and max_investment<=400:
        incr = 50
    else:
        incr = 100

    # setting domain for CSP variables
    domain = np.arange(0, max_investment + incr, incr)

    #creating variables
    variables = []
    for name in ds_names:
        variables.append(Variable(name, domain))


    #function for the volatility constraint
    def calculate_portfolio_volatility(*values):
        """
        :return: average volatility of the portfolio
        """

        # calculating the sum of weighted volatilities
        weighted_volatilities_sum = sum(
            value * asset_volatilities[asset] for value, asset in zip(values, asset_volatilities))

        # calculating the total sum of investments
        total_investment = sum(values)

        # checking for division by zero
        if total_investment == 0:
            return 0

        # calculating the portfolio volatility
        portfolio_volatility = weighted_volatilities_sum / total_investment

        return portfolio_volatility


    constraints = []

    # investment limits contraints
    constraints.append(Constraint(scope=variables, condition=lambda *values: sum(values) <= max_investment))
    constraints.append(Constraint(scope=variables, condition=lambda *values: sum(values) >= min_investment))

    # volatility contraint
    constraints.append(
        Constraint(scope=variables, condition=lambda *values: calculate_portfolio_volatility(*values) <= risk_factor))

    return CSP("Portfolio Optimization", variables, constraints)


def csp_solver(csp):
    """
    :param csp: Constraint Satisfaction Problem (CSP) instance to be solved.
    :return: List of all solutions found by the CSP solver.
    """

    # creating an instance of the Arc_Consistency solver for the given CSP
    arc_solver = Arc_Consistency(csp)

    # using the solver to find all solutions to the CSP
    all_solutions = arc_solver.solve_all_wrapper()

    return all_solutions



def load_model(dataset_name):
    """
    :param dataset_name: Name of the dataset for which the model is loaded.
    :return: Loaded model if found, otherwise returns None.
    """

    # defining the directory and filename for the model
    models_directory = 'models'
    model_filename = f'model_{dataset_name}.pkl'
    model_path = os.path.join(models_directory, model_filename)

    # checks if the model file exists
    if os.path.exists(model_path):
        # loads the model from the file
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        # prints a message if the model file is not found
        print(f"Model file not found for {dataset_name}")
        return None

def data_visualization(solution):
    # clears static directory and recreates it, so the files are not 'written over'

    #colors for time series
    colors = ['#6f3cff', '#fca778', '#c2bcff', '#7f50ff', '#af9cff', '#ffc282']

    #time series plot initialization
    fig = go.Figure()

    #lists for pie chart and polar chart
    assets = []
    allocations = []

    
    for var, y in solution.items():
        file_name = var.name

        #time series plot
        df = data_preparation('stocks/'+file_name)
        color = random.choice(colors)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['High'], mode='lines', name=file_name.replace('.csv', ''), marker=dict(color=color)))
        colors.remove(color)

        #for pie chart and polar chart
        file_name = file_name.replace(".csv", "")
        assets.append(file_name)
        allocations.append(y)


    #time series
    fig.update_layout(title='Assets trends',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      plot_bgcolor='#ffffff')

    # saves graph
    if os.path.exists("static/time_series_plot_interactive.html"):
        os.remove("static/time_series_plot_interactive.html")
    fig.write_html("static/time_series_plot_interactive.html")


    #pie chart
    colors = ['#6f3cff', '#fca778', '#c2bcff', '#ffc282', '#7f50ff']
    plt.pie(allocations, colors=colors, labels=assets)
    plt.axis('equal')

    # saves pie chart
    if os.path.exists("static/piegraph.jpg"):
        os.remove("static/piegraph.jpg")
    plt.savefig("static/piegraph.jpg", dpi=300)
    plt.cla()
    plt.clf()
    plt.close()

    # Polar chart
    theta = np.linspace(0, 2 * np.pi, len(assets), endpoint=False)

    plt.figure(figsize=(10, 6))
    plt.subplot(polar=True)


    # Plots allocations on polar chart
    plt.plot(theta, allocations)
    plt.fill(theta, allocations, 'b', alpha=0.2)

    # Adds legend and title
    plt.legend(labels=('Allocations',), loc=1)
    plt.title("Asset Allocations")

    # saves polar chart
    if os.path.exists("static/polargraph.jpg"):
        os.remove("static/polargraph.jpg")
    plt.savefig("static/polargraph.jpg", dpi=300)
    plt.cla()
    plt.clf()
    plt.close()


def get_features(dataset, X):
    """
    :param dataset: Name of the dataset.
    :param X: DataFrame containing the features.
    :return: DataFrame with only the features specified in the corresponding features file.
    """
    # Reads feature names from the features file
    with open(f'features/{dataset}_features.txt', 'r') as file:
        lines = file.readlines()

    # Create a list to store feature names from the file
    column_names_in_file = []

    # Populate the list with feature names from the file
    for line in lines:
        line = line.strip()
        column_names_in_file.append(line)

    # Creates a set of column names in the DataFrame X
    column_names_X = set(X.columns)

    # Drops columns from X that are not present in the features file
    for col in column_names_X:
        if col not in column_names_in_file:
            X = X.drop(columns=col, errors='ignore')

    return X


def main(min_investment, max_investment, risk_factor):
    # Suppresses warnings
    warnings.filterwarnings("ignore")

    # List of dataset names
    dataset_names = ["AMD", "CSCO", "QCOM", "SBUX", "TSLA"]

    # Appends '.csv' to each dataset name
    datasets = [dataset + '.csv' for dataset in dataset_names]

    # Calculates volatilities for assets
    asset_volatilities = calculate_volatility(datasets)

    # Dictionary to store prediction data for each dataset
    prediction_data = {}

    # Dictionary to store the last 'Close' values of the training set for each dataset
    last_y_train_values = {}

    # Dictionary to store the last 'Close' values of the training set for each dataset
    datasets = []

    for dataset_name in dataset_names:
        # Load or train and load the model for each dataset
        model = load_model(dataset_name)
        dataset = dataset_name+'.csv'

        if model is not None:
            print(f"Model loaded successfully for {dataset_name}")
        else:
            modelling(dataset)
            model = load_model(dataset_name)
            print(f"Model loaded successfully for {dataset_name}")

        # Append dataset name to the list
        datasets.append(dataset)

        # Feature engineering
        df = feature_eng(dataset)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna()
        df.set_index('Date', inplace=True)

        # Splits the dataset into training and test sets
        split_date = df.index.max() - timedelta(days=365)
        train = df[df.index <= split_date]
        test = df[df.index > split_date]

        # Separate independent variables (X) from dependent variables (Y)
        X_train = train.drop('Close', axis=1)
        y_train = train['Close']
        # We need the same features that the model has been trained on
        X_test = get_features(dataset_name, test)


        # Makes predictions on the test set (future data)
        y_pred = model.predict(X_test)

        # Records the last 'Close' value of the training set for each dataset
        last_y_train_values[dataset] = y_train.iloc[-1]

        # Records the last 'Close' value of the prediction set for each dataset
        prediction = y_pred[-1]
        prediction_data[dataset] = prediction

    # Builds the CSP for portfolio optimization
    csp = build_portfolio_csp(datasets, min_investment, max_investment, risk_factor, asset_volatilities)

    # Solve the CSP to obtain all possible portfolio solutions
    solutions = csp_solver(csp)

    # Finds the best portfolio solution with the highest expected return
    best_return = 0
    best_solution = None
    for solution in solutions:
        #initializing total return of the portfolio
        total_return = 0

        # Calculates expected return for each asset in the portfolio
        for variable, allocation in solution.items():
            dataset_name = variable.name
            prediction = prediction_data[dataset_name]
            last_asset_value = last_y_train_values[dataset_name]

            #represents an estimate of the return expected from the current asset
            expected_return = prediction * (allocation / last_asset_value) - allocation

            total_return += expected_return

        #picks the solution with the highest expected return
        if total_return > best_return:
            best_return = total_return
            best_solution = solution

    # Visualizes and prints the best portfolio solution if found
    if best_solution is not None:
        final_solution = {}
        for x, y in best_solution.items():
            if y!=0:
                final_solution[x] = y


        data_visualization(final_solution)


        best_solution_str = {key: round(value, 2) for key, value in final_solution.items()}
        print(str(best_solution_str) + '. Expected return: ' + str(best_return))


        return final_solution

    # Returns None if no optimal solution is found
    return None

