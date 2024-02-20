import warnings

# suppresses warnings
warnings.filterwarnings("ignore")

import pandas as pd
import ta
from sklearn.neighbors import KNeighborsRegressor
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def data_preparation(dataset_name):
    """
    :param dataset_name: takes a csv file name as an input
    :return: ready to use dataframe
    """

    #converts dataset into dataframe
    df = pd.read_csv(dataset_name)

    # converting dates in yyyy-mm-dd format
    df['Date'] = pd.to_datetime(df['Date'])

    # renamed the column
    df.rename(columns={'Close/Last': 'Close'}, inplace=True)

    # converted to purely numeric values
    df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)
    df['Open'] = df['Open'].replace('[\$,]', '', regex=True).astype(float)
    df['High'] = df['High'].replace('[\$,]', '', regex=True).astype(float)
    df['Low'] = df['Low'].replace('[\$,]', '', regex=True).astype(float)


    # filling in missing values
    df = df.ffill()
    return df




def feature_eng(ds_name):
    """
    :param ds_name: takes a csv file name as an input
    :return: dataframe which includes new significant values
    """

    #building path to dataset
    ds_name = 'stocks/' + ds_name

    # pre-processing data
    df = data_preparation(ds_name)


    # Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Close'])
    # On Balance Volume (OBV)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    # Take Profit
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Simple Moving Average
    # 14 days
    df['sma14'] = ta.trend.sma_indicator(df['Close'], window=14)
    # 30 days
    df['sma30'] = ta.trend.sma_indicator(df['Close'], window=30)
    # 100 days
    df['sma100'] = ta.trend.sma_indicator(df['Close'], window=100)

    # Mean Absolute Deviation
    df['mad'] = df['TP'].rolling(14).apply(lambda x: (pd.Series(x) - pd.Series(x).mean()).abs().mean())
    # Commodity Channel Index (30 days window)
    df['CCI'] = (df['TP'] - df['sma30']) / (0.015 * df['mad'])

    # Moving Average Convergence-Divergence
    # Calculating Exponential Moving Average  for 12 and 26 periods
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    # Calculating Moving Average Convergence-Divergence (MACD)
    df['macd'] = df['ema_12'] - df['ema_26']


    # drops na values (caused by values calculated over a window of time)
    df = df.dropna()


    return df


def feature_selection(X_train, y_train, X_test):
    """
    takes train and test groups as inputs
    :return: the X groups (train and test) with only selected features, and name of the selected features
    """

    # we'd prefer not to count banal columns
    columns_to_drop = ['Open', 'High', 'Low']
    X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
    X_test = X_test.drop(columns=columns_to_drop, errors='ignore')

    model = LinearRegression()

    # initializating rfe for feature selection
    rfe = RFE(model, n_features_to_select=5)

    # training and transforming
    X_train_selected = rfe.fit_transform(X_train, y_train)
    X_test_selected = rfe.transform(X_test)

    # giving feature names to rfe model
    rfe_feature_names = X_train.columns[rfe.support_]


    return X_train_selected, X_test_selected, rfe_feature_names


def accuracy_scatter(y_test, y_pred, dataset_name):
    """
    :return: a scatter plot which illustrates how accurate are the predicted values
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title(f'Scatter Plot for  {dataset_name}')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.grid(True)

    # Saves scatter plot in the performance directory
    performance_directory = 'performance'
    save_path = os.path.join(performance_directory, f'{dataset_name}_accuracy_plot.png')
    plt.savefig(save_path, format='png')

def choose_model(X_train, y_train, X_test, y_test, ds_name):
    """returns the best model for the data set, based on the MSE score"""

    #picks between linear regressor, random forest, and k neighbors regressor
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'K Neighbors Regressor' : KNeighborsRegressor(),
    }

    model = None
    best_mse = float('inf')

    # creates a new directory if it does not exist, to contain performance files
    performance_directory = 'performance'
    os.makedirs(performance_directory, exist_ok=True)

    # to write print statements directly into performance file
    with open(f'{performance_directory}/{ds_name}_performances.txt', 'w') as file:
        # iterates over the models
        for name, possible_model in models.items():
            # fits the model to the train groups
            possible_model.fit(X_train, y_train)
            # predicts the target values of the x_test group
            y_pred = possible_model.predict(X_test)
            # calculates accuracy of predictions using mean squared error
            mse = mean_squared_error(y_test, y_pred)

            # finding model with lowest mse
            if mse < best_mse:
                best_mse = mse
                model = possible_model
                # writes in the file which model was chosen
                print(f"For the dataset {ds_name}, {name} was chosen.", file=file)
        #draws accuracy scatter for selected model
        accuracy_scatter(y_test, y_pred, ds_name)
    return model


def modelling(dataset):
    """
    function that trains the models
    """

    #performs feature engineering
    df = feature_eng(dataset)

    # sets date as index
    df.set_index('Date', inplace=True)

    # close price is target value
    X = df.drop('Close', axis=1)
    Y = df['Close']

    # splitting dataset for feature selection
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #performs feature selection
    X_train, X_test, features = feature_selection(X_train, y_train, X_test)

    #saving the features in a txt file
    # (if the model is linear regressor it does not have a "feature_name" attribute)
    os.makedirs('features', exist_ok=True)
    dataset_name = dataset.replace('.csv', '')
    with open(f'features/{dataset_name}_features.txt', 'w') as file:
        for name in features:
            print(f"{name}", file=file)


    # splitting train group for cross validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # needed dataset names without extensions to name .pkl and .txt files
    dataset_name = os.path.splitext(dataset)[0]

    # choosing model
    model = choose_model(X_train, y_train, X_test, y_test, dataset_name)

    #indicating feature_names
    model.feature_names = features

    # training the model
    model.fit(X_train, y_train)
    # predicting
    y_pred = model.predict(X_test)

    # calculating mse
    mse = mean_squared_error(y_test, y_pred)
    # calculating r2 scored
    r2 = r2_score(y_test, y_pred)
    # calculating residual standard deviation
    std_residual = np.sqrt(mse)

    # cross_validation
    cross_val_scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
    # calculating average cross validation
    avg_cross_val_mse = -cross_val_scores.mean()

    # saving performance data
    with open(f'performance/{dataset_name}_performances.txt', 'a') as file:
        print("STATISTICS:", file=file)
        print(
            f"MSE: {mse}\nR²: {r2}\nResidual Standard Deviation: {std_residual}\nAverage mse with cross validation: {avg_cross_val_mse}",
            file=file)

    # if it doesn't exist, creates new directory to contain the pre - trained models
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    #saving models in a directory
    with open(f'{models_directory}/model_{dataset_name}.pkl', 'wb') as file:
        # dumps pre-trained model in the .pkl file
        pickle.dump(model, file)
    print(f"Saved and trained model for {dataset}!")


# runs only if running directly from this script. Only running once to have pretrained models should suffice.
if __name__ == "__main__":
    datasets = ["AMD.csv", "CSCO.csv", "QCOM.csv", "SBUX.csv", "TSLA.csv"]
    for dataset in datasets:
        modelling(dataset)

