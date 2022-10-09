#### Model Load
import pickle
import pandas as pd
import streamlit as st
import datetime as dt

#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder, StandardScaler
#from sklearn.compose import ColumnTransformer

import talib as ta
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

#import warnings
#warnings.simplefilter(action='ignore')

input_file_myr = f'model/lgbm_myr.bin'
input_file_usd = f'model/lgbm_usd.bin'

class Forecasting:
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess
        self.data = None
        self.date_prediction = None

    #@st.experimental_memo
    def get_data(self, currency, start, end, interval="1d"):
        """
        fetch data from yahoo finance then do simple data cleaning
        """
        data = pdr.get_data_yahoo(currency, start=start, end=end, interval=interval)
        if data.empty:
            st.error('Data fetch unsuccessfull', icon="🚨")
    
        close = data.drop(columns=["Volume", "Adj Close"])
        close = close.fillna(method= 'ffill')
        close.sort_index(inplace=True)
        self.data = close
        
    def create_features(self, window=3):
        """
        Features engineering
        """
        df = self.data.copy()

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
        df["close_high"] = df["Close"] - df["High"]
        df["close_low"] = df["Close"] - df["Low"]

        for i in range(1, window+1):
            for col in df.loc[:, "upper_band":"close_low"].columns:
                df[col+ "_lag_" + str(i)] = df[col].shift(i)  
        
        df.drop(columns=["Close", "Open", "High", "Low"], inplace=True)

        self.data = df

    def predict(self):
        label = {0: 'Model expects GBP to appreciate so buy', 1: 'Model expects GBP to depreciate so sell'}
        input = pd.DataFrame(self.data.iloc[-1,:]).T
        X = self.preprocess.transform(input) #get today's date for prediction
        y = self.model.predict(X)[0] #predict return a a label inside a list
        return label[y]


with open(input_file_myr, "rb") as f_in:
    preprocess_myr, model_myr = pickle.load(f_in)

with open(input_file_usd, "rb") as f_in:
    preprocess_usd, model_usd = pickle.load(f_in)

end = dt.date.today()
start = end - dt.timedelta(days=30)

st.header("*Machine Learning* Currency Prediciton :sunglasses:")

option = st.selectbox(
    "Select Currency",
    ("GBPUSD", "GBPMYR"))

st.metric(label="Prediction's Date", value = str(end+dt.timedelta(days=1)))

model_gbpusd = Forecasting(model_usd, preprocess_usd)
model_gbpmyr = Forecasting(model_myr, preprocess_myr)

if (end.weekday() == 5) or (end.weekday() == 4):
    st.error("Market is closed tomorrow. No Prediction is available", icon="🚨")
else:
    if option == "GBPUSD":
        with st.spinner("Fetching data and Making Prediction...."):
            model_gbpusd.get_data(str(option+ "=X"), start, end)
            model_gbpusd.create_features()
            prediction = model_gbpusd.predict()
    else:
        with st.spinner("Fetching data and Making Prediction...."):
            model_gbpmyr.get_data(str(option+ "=X"), start, end)
            model_gbpmyr.create_features()
            prediction = model_gbpmyr.predict()
    st.write(prediction)
    


