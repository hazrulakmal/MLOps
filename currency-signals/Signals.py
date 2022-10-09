
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import talib as ta
from talib import abstract
import random

import warnings
warnings.simplefilter(action='ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

import pickle

output_file = f'model/xgb_model.bin'


def create_target(df, auto=True, upper=None, lower=None, quantile=0.75):
    
    def map_target(pip):
        if pip > upper:
            return "buy"
        elif pip < lower:
            return "sell"
        else:
            return "hold"
        
    def up_down(pip):
        if pip > 0:
            return "buy"
        else:
            return "sell"
        
    row_random = random.choice([i for i in range(0, len(df))])
    check_decimals = len(str(df.loc[row_random, "Close"]).split(".")[1])
    
    #currency and commodaties have different decimal places for calculating pips
    if check_decimals == 2:
        pip_multiplier = 10
    else:
        pip_multiplier = 10**4
    
    df["pips"] = df["Close"].diff().shift(-1) * pip_multiplier
    df.dropna(inplace=True)
    
    if auto == True:
        upper = df["pips"].quantile(quantile)
        lower = df["pips"].quantile(1-quantile)
    
    df["Target"] = df["pips"].apply(up_down)
    df.drop(columns="pips", inplace=True)
    
    #print(f"upper: {upper}")
    #print(f"lower: {lower}")
    return df.set_index("Date")


data = pd.read_csv("src/currency_exchange_updated.csv", date_parser=["Date"])
data["Date"] = pd.to_datetime(data["Date"])

#get XAUUSD data
gold = data[data["Currency"] == "XAU_USD"]
print(f"Row : {gold.shape[0]}")
gold.drop(columns="Currency",inplace=True)
gold.sort_values(by="Date", inplace=True)

gold_df = create_target(gold, quantile=0.75)


le = LabelEncoder() #re-classing the target
gold_df["Target"] = le.fit_transform(gold_df["Target"])

mapping = {}
for each_class in le.classes_:
    mapping[int(le.transform([each_class]))] = each_class


# ## Feature Engineering
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
        for col in df.drop(columns="Target").loc[:, "upper_band":"close_open"].columns:
            df[col+ "_lag_" + str(i)] = df[col].shift(i)  

    return df

featured_df = create_technical_indicators(gold_df)
featured_df.dropna(inplace=True)


#### Candlesticks

def create_candlesticks(df):
    df.rename(columns={'Open': 'open', 'High': 'high','Low': 'low','Close': 'close'}, inplace= True)
    
    #candlestick_features
    candlesticks = ta.get_function_groups()["Pattern Recognition"]
    for indicator in candlesticks:
        df[str(indicator)] = getattr(abstract, indicator)(df)

    #remove less-common candlestick
    removed = []
    for candle in candlesticks:
        non_detected = df[candle].value_counts()[0]
        if non_detected > 4000:
            removed.append(candle)
            df.drop(columns=candle, inplace=True)
    
    attr_cat = list(set(candlesticks) - set(removed))
    return df, attr_cat

#engineered_features, attr_cat = create_candlesticks(featured_df)
#new_predictors = list(set(featured_df.columns) - set(["open", "low", "high", "close", "Target"]))
#attr_num = list(set(new_predictors)-set(attr_cat))


### Model

## 1 - categorical pipeline
#cat_pipeline = Pipeline(steps = [
#    ('encode', OneHotEncoder(handle_unknown='ignore'))
#])
attr_num = list(set(featured_df.columns) - set(["open", "low", "high", "close", "Target"]))

num_pipeline = Pipeline([
    ('scale', StandardScaler())
])

preprocessing_pipeline = ColumnTransformer([
    #('cat', cat_pipeline, attr_cat),
    ('num', num_pipeline, attr_num)
])

new_predictors = list(set(featured_df.columns) - set(["open", "low", "high", "close", "Target"]))

train = preprocessing_pipeline.fit_transform(featured_df[new_predictors])
y_train = featured_df["Target"]


#### Validation
print("Validation in progress")
model = XGBClassifier(random_state=51, verbosity = 0)

def val_score(model, X_train, y_train, n_splits=10):
    kf = TimeSeriesSplit(n_splits=n_splits, gap=0)
    cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring= "accuracy")
    print(f" Accuracy Score : {round(cv_results.mean(), 5)} +- ({round(cv_results.std(), 5)}) {cv_results}")
    
val_score(model, train, y_train)


print("Model in Training")
model.fit(train, y_train)
#predictions = model.predict(test)
#results =pd.DataFrame({"y_true": y_test, "preds": predictions})
#accuracy_score(y_test,  predictions)

#### Model Save
with open(output_file, 'wb') as f_out: #wb write and binary
    pickle.dump((preprocessing_pipeline, model), f_out)
print("Model Saved")


