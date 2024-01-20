#importing libraries

from datetime import datetime, date, timedelta
import pandas as pd
import ta
import numpy as np
import statistics
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error




#to change with the desired dataset
df = pd.read_csv("AAPL.csv")

"""
------ Data preparation ------
"""

#converting dates in yyyy-mm-dd format
df['Date'] = pd.to_datetime(df['Date'])


#renamed the column for a pure personal preference
df.rename(columns={'Close/Last': 'Close'}, inplace=True)

#converted to purely numeric values
df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)
df['Open'] = df['Open'].replace('[\$,]', '', regex=True).astype(float)
df['High'] = df['High'].replace('[\$,]', '', regex=True).astype(float)
df['Low'] = df['Low'].replace('[\$,]', '', regex=True).astype(float)

df = df.ffill()

"""

--- FEATURE ENGINEERING ---


"""

Buy_date="10/21/2021"
span=100
targetdate=datetime.strptime(Buy_date,"%m/%d/%Y")

"""
RAPPORTI DI RITORNO COMULATIVI, TARGET:3g,7g,14g
"""

targetdate_3g=targetdate+timedelta(days=3)
targetdate_7g=targetdate+timedelta(days=7)
targetdate_14g=targetdate+timedelta(days=14)

df_filtered_base = df[df['Date'] == pd.to_datetime(targetdate)]
df_filtered_3g = df[df['Date'] == pd.to_datetime(targetdate_3g)]
df_filtered_7g = df[df['Date'] == pd.to_datetime(targetdate_7g)]
df_filtered_14g = df[df['Date'] == pd.to_datetime(targetdate_14g)]

if df_filtered_base.empty:
    print("No data for targetdate")
else:

    close_base = df_filtered_base['Close'].values[0]
    if df_filtered_3g.empty:
        print("Nessun dato per Ritorno cumulato a 3 giorni")
    else:
        close_3g = df_filtered_3g['Close'].values[0]
        print("Ritorno cumulato a 3 giorni:{:.2f}% ".format(((close_3g-close_base)/close_base)*100))
    if df_filtered_7g.empty:
        print("Nessun dato per Ritorno cumulato a 7 giorni")
    else:
        close_7g = df_filtered_7g['Close'].values[0]
        print("Ritorno cumulato a 7 giorni:{:.2f}% ".format(((close_7g-close_base)/close_base)*100))
    if df_filtered_14g.empty:
        print("Nessun dato per Ritorno cumulato a 14 giorni")
    else:
        close_14g = df_filtered_14g['Close'].values[0]
        print("Ritorno cumulato a 14 giorni:{:.2f}%".format(((close_14g-close_base)/close_base)*100))


"""
STUDIO SUI RAPPORTI DI RITORNO
"""

df_ROI = pd.DataFrame()


df_100g = df[df['Date'] >= pd.to_datetime(targetdate)]
df_100g = df_100g[df_100g['Date'] <= pd.to_datetime(targetdate) + timedelta(days=100)]

Buy_date_close=df[df['Date'] == pd.to_datetime(targetdate)]['Close'].values[0]

for date in df_100g.iterrows():
    close = date[1]['Close']
    ROI= (close/Buy_date_close)*100
    df_ROI=pd.concat([df_ROI,pd.DataFrame({'Date': [date[1]['Date']], 'ROI': [ROI]})])
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
VOLATILITY
"""
#calculating relative price: proportional change in stock price between a day and the dau before
df['Price relative'] = ""
for i in range(1, len(df.Date)):
    df.loc[i, 'Price relative'] = df['Close'][i] / df['Close'][i - 1]

#calculating proportional change in stock price between a day and the dau before
df['Daily Return'] = ""
for i in range(1, len(df.Date)):
    df.loc[i, 'Daily Return'] = np.log(df['Close'][i] / df['Close'][i - 1])

#daily volatility
DailyVolatility = statistics.stdev(df['Daily Return'][1:])
print("The daily volatility  is: {:.2%}".format(DailyVolatility))


#annulized daily voltility
AnnualizedDailyVolatilityCalendarDays = DailyVolatility*np.sqrt(365)
print("The annualized daily volatility measured in calendar days is: {:.2%}".format(AnnualizedDailyVolatilityCalendarDays))

df = df.drop('Daily Return', axis=1)
df = df.drop('Price relative', axis=1)

"""
MOVING AVERAGE CONVERGENCE-DIVERGENCE
"""
df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

df['macd'] = df['ema_12'] - df['ema_26']

df.drop('ema_12', axis=1, inplace=True)
df.drop('ema_26', axis=1, inplace=True)


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
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna()
df.set_index('Date', inplace = True)

X = df.drop('Close', axis=1)
Y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Machine': SVR()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - MSE: {mse}")
  