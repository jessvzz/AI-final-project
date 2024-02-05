from datetime import datetime, date, timedelta
import pandas as pd
import ta
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
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

"""

--- FEATURE ENGINEERING ---


"""

def feature_eng(ds_name):
    ds_name = 'stocks/'+ds_name
    df = data_preparation(ds_name)
    Buy_date = "10/21/2021"
    span = 100
    targetdate = datetime.strptime(Buy_date, "%m/%d/%Y")

    """
    STUDIO SUI RAPPORTI DI RITORNO
    """

    df_ROI = pd.DataFrame()

    df_100g = df[df['Date'] >= pd.to_datetime(targetdate)]
    df_100g = df_100g[df_100g['Date'] <= pd.to_datetime(targetdate) + timedelta(days=100)]

    Buy_date_close = df[df['Date'] == pd.to_datetime(targetdate)]['Close'].values[0]

    for date in df_100g.iterrows():
        close = date[1]['Close']
        ROI = (close / Buy_date_close) * 100
        df_ROI = pd.concat([df_ROI, pd.DataFrame({'Date': [date[1]['Date']], 'ROI': [ROI]})])
        df_ROI['ROI_Moving_Avg'] = df_ROI['ROI'].rolling(window=3).mean()

    """
    INDICATORI TECNICI
    """
    df_ema = pd.DataFrame()
    df_ema['Date'] = df['Date']
    df_ema['Close'] = df['Close']
    df_ema['EMA'] = df['Close'].ewm(span=100, adjust=False).mean()
    df_ema['RSI'] = ta.momentum.rsi(df['Close'])

    """
    ON BALANCE VOLUME
    """
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

    """
    TAKE-PROFIT
    """
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3

    """
    SIMPLE MOVING AVERAGE
    """
    df['sma'] = df['TP'].rolling(14).mean()

    """
    MEAN ABSOLUTE DEAVIATION
    """
    df['mad'] = df['TP'].rolling(14).apply(lambda x: (pd.Series(x) - pd.Series(x).mean()).abs().mean())

    """
    COMMODITY CHANNEL INDEX
    """
    df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])



    """
    MOVING AVERAGE CONVERGENCE-DIVERGENCE
    """
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    df['macd'] = df['ema_12'] - df['ema_26']

    df.drop('ema_12', axis=1, inplace=True)
    df.drop('ema_26', axis=1, inplace=True)
    return df


"""

--- FEATURES SELECTION ---

df = df.dropna()

features = df.columns.tolist()
features.remove('Close')
features.remove('Date')

x = df[features]
y = df['Close']

model = RandomForestRegressor()

model.fit(x, y)
feature_importance = model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)
"""



"""

--- SPLITTING THE DATASET --- 

"""
def choose_model(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Support Vector Machine': SVR()
    }

    model = None
    best_mse = float('inf')

    for name, possible_model in models.items():
        possible_model.fit(X_train, y_train)
        y_pred = possible_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} - MSE: {mse}")

        if mse < best_mse:
            best_mse = mse
            model = possible_model
    return model

def modelling(dataset):
        df = feature_eng(dataset)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna()
        df.set_index('Date', inplace=True)

        X = df.drop('Close', axis=1)
        Y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        model = choose_model(X_train, y_train, X_test, y_test)

        model.fit(X, Y)

        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        std_residual = np.sqrt(mse)

        print(f"MSE: {mse}, R²: {r2}, Deviazione Standard Residua: {std_residual}")

        # cross_validation
        cross_val_scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        avg_cross_val_mse = -cross_val_scores.mean()

        print(f"Media MSE con validazione incrociata: {avg_cross_val_mse}")
        """
        dataset_name = os.path.splitext(dataset)[0]
        models_directory = 'models'

        os.makedirs(models_directory, exist_ok=True)

        with open(f'{models_directory}/model_{dataset_name}.pkl', 'wb') as file:
            pickle.dump(model, file)
        print(f"Saved and trained model for {dataset}!")

if __name__ == "__main__":
    datasets = ["AMD.csv", "CSCO.csv", "QCOM.csv", "SBUX.csv", "TSLA.csv"]
    for dataset in datasets:
        modelling(dataset)

