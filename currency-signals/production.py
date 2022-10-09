#### Model Load
import pickle
import talib as ta
import pandas as pd

import warnings
warnings.simplefilter(action='ignore')

def create_technical_indicators(df_original, window=3):
    df = df_original.copy()
    
    #Bollinger Bands
    df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['Close'], timeperiod =20)
    
    #Stochastic Oscillators
    df['slowk'], df['slowd'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    #RSI
    df['rsi_14'] = ta.RSI(df['Close'], 14)

    #Absolute Price Osicillator
    df["apo"] = ta.APO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    
    #Exponential Moving Average
    df["ema_7"] = ta.EMA(df['Close'], timeperiod=7)
    
    #ratios
    df["ema_midband"] = df["middle_band"] / df["ema_7"]
    df["close_ema"] = df["Close"] / df["ema_7"]
    df["close_upband"] = df["Close"] / df["upper_band"]
    df["close_lowband"] = df["Close"] / df["lower_band"]
    df["close_midband"] = df["Close"] / df["middle_band"]
    df["slowk_slowd"] = df['slowk'] / df['slowd']
    df["rsi_pct_change"] = df["rsi_14"].pct_change()
    df["rsi_stoch"] = df["rsi_14"] / ((df['slowk'] + df['slowd'])/2)
    
    df["close_open"] = df["Close"] - df["Open"]
    
    for i in range(1, window+1):
        for col in df.loc[:, "upper_band":"close_open"].columns:
            df[col+ "_lag_" + str(i)] = df[col].shift(i)  

    return df

input_file = f'model/xgb_model.bin'

with open(input_file, "rb") as f_in:
    preprocess, model = pickle.load(f_in)

data =  pd.read_csv("src/currency_exchange_updated.csv", date_parser=["Date"])
data["Date"] = pd.to_datetime(data["Date"])
#get XAUUSD data
gold = data[data["Currency"] == "XAU_USD"]
print(f"Row : {gold.shape[0]}")
gold.drop(columns="Currency",inplace=True)
gold.sort_values(by="Date", inplace=True)

last_30 = gold.iloc[-32:,:]
featured_df = create_technical_indicators(last_30)
featured_df.dropna(inplace=True)
print("features extracted")

new_predictors = list(set(featured_df.columns) - set(["open", "low", "high", "close"]))
test = featured_df.iloc[-1:, :]

#Prediction
X = preprocess.transform(test[new_predictors])
print(f"prediction on {str(test.Date)}")

print(f"Classification: {model.predict(X)}")
