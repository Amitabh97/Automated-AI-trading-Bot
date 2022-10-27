
# *Automated AI Trading Bot* 

## _Goal_ : 
In this project I have created two trading models, one using deep learning and other using combination of strategies (SMA + Bollinger band), combined the performance of both the strategies and then automated the more profitable model by deploying it in AWS.

## ***_Disclaimer: Please don't trade without experience/financial knowledge. It can cost a lot!_

## Platform used: 
FXCM is used as trading platform for performing the algo trades

![App Screenshot](https://www.strategie-bourse.com/media/images/trading.jpg)
# Deep Learning Model: 
In this model, I have created a deep learning trading system using RNN and backtested the model and evaluated it’s performance over time.

### Step 1: Getting and preparing data – 
A) _Import the following libraries_:

```bash
import pandas as pd
import numpy as np
import fxcmpy
import matplotlib.pyplot as plt
plt.style.use("seaborn")
pd.set_option('display.float_format', lambda x: '%.5f' % x) 

```

B) _Getting EUR/USD data from FXCM(limit=10000 candles)_
```bash
api = fxcmpy.fxcmpy(config_file= "fxcm.cfg")
data = api.get_candles('EUR/USD', start = "2020-12-01", end = "2022-07-27", 
                period = "H1")
```
C) _Converting to DF and saving in CSV file_
```bash
df=pd.DataFrame(data)
df.to_csv('file.csv')
data_csv = pd.read_csv('file.csv', parse_dates = ["date"], index_col = "date", usecols = ['date','bidclose'])
data_csv.rename(columns = {'bidclose':'price'}, inplace = True)
symbol = data_csv.columns[0]
data_csv["returns"] = np.log(data_csv[symbol] / data_csv[symbol].shift())
```
### Step 2: Now features are added to the dataset on which model will be build
```bash
window = 50
df = data_csv.copy() # saving copy of data as df
df["dir"] = np.where(df["returns"] > 0, 1, 0) # if return >0 ->1, else 0 (earlier it was -1,0,1 but now as working with DNN, so only two o/p 0/1)....it is also a feature for DNN
df["sma"] = df[symbol].rolling(window).mean() - df[symbol].rolling(150).mean() # 2nd feature is distance between ->SMA 50,150
df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()) / df[symbol].rolling(window).std() # 3rd feature is distance between price and (SMA mean/SMA std)
df["min"] = df[symbol].rolling(window).min() / df[symbol] - 1 # min SMA price in terms of current price as percentage
df["max"] = df[symbol].rolling(window).max() / df[symbol] - 1 # max SMA price in terms of current price as percentage
df["mom"] = df["returns"].rolling(3).mean() # momentum feature taking mean of last 3 candles
df["vol"] = df["returns"].rolling(window).std() # volatility feature
df.dropna(inplace = True)
lags = 5
cols = []
features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
for f in features:
        for lag in range(1, lags + 1):
            col = "{}_lag_{}".format(f, lag)
            df[col] = df[f].shift(lag)
            cols.append(col)
df.dropna(inplace = True)
```
### Step 3: Split the data into train and test set
```bash
split = int(len(df)*0.75)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()
```
### Step 4: Feature Scaling – Standardization is done
```bash
mu, std = train.mean(), train.std()
train_s = (train - mu) / std
```
### Step 5: Creating and Fitting the DNN Model
_Creating the DNN model_ – 
```bash
import random
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam

def set_seeds(seed = 100): # setting same seed ensures we get same model as output
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def cw(df): # class weight
    c0, c1 = np.bincount(df["dir"])
    w0 = (1/c0) * (len(df)) / 2
    w1 = (1/c1) * (len(df)) / 2
    return {0:w0, 1:w1}

optimizer = Adam(lr = 0.0001) # adam optimiser with learning rate

def create_model(hl = 2, hu = 100, dropout = False, rate = 0.3, regularize = False,
                 reg = l1(0.0005), optimizer = optimizer, input_dim = None):
    if not regularize:
        reg = None
    model = Sequential()
    model.add(Dense(hu, input_dim = input_dim, activity_regularizer = reg ,activation = "relu"))
    if dropout: 
        model.add(Dropout(rate, seed = 100))
    for layer in range(hl):
        model.add(Dense(hu, activation = "relu", activity_regularizer = reg))
        if dropout:
            model.add(Dropout(rate, seed = 100))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    return model
Now importing this model – 
from DNNModel import *
set_seeds(100)
model = create_model(hl = 3, hu = 50, dropout = True, input_dim = len(cols))
model.fit(x = train_s[cols], y = train["dir"], epochs = 50, verbose = False,
          validation_split = 0.2, shuffle = False, class_weight = cw(train))
model.evaluate(train_s[cols], train["dir"])
pred = model.predict(train_s[cols])
```
### Step 6: Forward testing the model
```bash
model.evaluate(test_s[cols], test["dir"])
pred = model.predict(test_s[cols])
test["prob"] = model.predict(test_s[cols])
test["position"] = np.where(test.prob < 0.47, -1, np.nan) # 1. short where proba < 0.47
test["position"] = np.where(test.prob > 0.53, 1, test.position) # 2. long where proba > 0.53
test.index = test.index.tz_localize("UTC")
test["NYTime"] = test.index.tz_convert("America/New_York")
test["hour"] = test.NYTime.dt.hour
test["position"] = np.where(~test.hour.between(2, 12), 0, test.position) # 3. neutral in non-busy hours
test["position"] = test.position.ffill().fillna(0) # 4. in all other cases: hold position
test["strategy"] = test["position"] * test["returns"]
test["creturns"] = test["returns"].cumsum().apply(np.exp)
test["cstrategy"] = test["strategy"].cumsum().apply(np.exp)
ptc = 0.000059 # trading cost
test["trades"] = test.position.diff().abs()
test["strategy_net"] = test.strategy - test.trades * ptc
test["cstrategy_net"] = test["strategy_net"].cumsum().apply(np.exp)
test[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12, 8))
plt.show()
```
### Step 7: Now we save the model and it’s parameters for live trading
```bash
model.save("DNN_model")
import pickle
params = {"mu":mu, "std":std}
pickle.dump(params, open("params.pkl", "wb")) # saving the dict
```
### Step 8: Implementing the model in live market
__A) Import libraries__
```bash
import pandas as pd
import numpy as np
import fxcmpy
from datetime import datetime, timedelta
import time
```
__B)	Load model and parameters__
```bash
import keras
model = keras.models.load_model("DNN_model")
import pickle
params = pickle.load(open("params.pkl", "rb"))
mu = params["mu"]
std = params["std"]
```
__C)	[FXCM] Implementation__
```bash
api = fxcmpy.fxcmpy(config_file= "fxcm.cfg")
col = ["tradeId", "amountK", "currency", "grossPL", "isBuy"]
class DNNTrader():
    
    def __init__(self, instrument, bar_length, window, lags, model, mu, std, units):
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length) 
        self.tick_data = None
        self.raw_data = None
        self.data = None 
        self.ticks = 0
        self.last_bar = None  
        self.units = units
        self.position = 0
        
        #*****************add strategy-specific attributes here******************
        self.window = window
        self.lags = lags
        self.model = model
        self.mu = mu
        self.std = std
        #************************************************************************        
    
    def get_most_recent(self, period = "m1", number = 10000):
        while True:  
            time.sleep(5)
            df = api.get_candles(self.instrument, number = number, period = period, columns = ["bidclose", "askclose"])
            df[self.instrument] = (df.bidclose + df.askclose) / 2
            df = df[self.instrument].to_frame()
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()) - self.last_bar < self.bar_length:
                self.start_time = pd.to_datetime(datetime.utcnow()) # NEW -> Start Time of Trading Session
                break
    
    def get_tick_data(self, data, dataframe):
        
        self.ticks += 1
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(data["Updated"], unit = "ms")
        
        if recent_tick - self.last_bar > self.bar_length:
            self.tick_data = dataframe.loc[self.last_bar:, ["Bid", "Ask"]]
            self.tick_data[self.instrument] = (self.tick_data.Ask + self.tick_data.Bid)/2
            self.tick_data = self.tick_data[self.instrument].to_frame()
            self.resample_and_join()
            self.define_strategy() 
            self.execute_trades()
            
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                             label="right").last().ffill().iloc[:-1])
        self.last_bar = self.raw_data.index[-1]  
        
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        df = df.append(self.tick_data.iloc[-1]) # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["dir"] = np.where(df["returns"] > 0, 1, 0)
        df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
        df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace = True)
        
        # create lags
        self.cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                self.cols.append(col)
        df.dropna(inplace = True)
        
        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        df["proba"] = self.model.predict(df_s[self.cols])
        
        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position"] = np.where(df.proba < 0.47, -1, np.nan)
        df["position"] = np.where(df.proba > 0.53, 1, df.position)
        df["position"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING LONG")  
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING SHORT")  
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING SHORT")  
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL") 
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL")  
            self.position = 0

    def report_trade(self, order, going):
        time = order.get_time()
        units = api.get_open_positions().amountK.iloc[-1]
        price = api.get_open_positions().open.iloc[-1]
        unreal_pl = api.get_open_positions().grossPL.sum()
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | Unreal. P&L = {}".format(time, units, price, unreal_pl))
        print(100 * "-" + "\n")
trader = DNNTrader("EUR/USD", bar_length = "20min", 
                   window = 50, lags = 5, model = model, mu = mu, std = std, units = 100)
trader.get_most_recent()
api.subscribe_market_data(trader.instrument, (trader.get_tick_data, ))
api.unsubscribe_market_data(trader.instrument)
if len(api.get_open_positions()) != 0: # if we have final open position(s) (netting and hedging)
    api.close_all_for_symbol(trader.instrument)
    print(2*"\n" + "{} | GOING NEUTRAL".format(str(datetime.utcnow())) + "\n")
    time.sleep(20)
    print(api.get_closed_positions_summary()[col])
    trader.position = 0
trader.data
api.close()
```
# Combination of Strategies:

We have already performed data collection and pre-processing in above steps
With the combination of SMA and Bollinger band, I have created two strategies:

__Strategy 1 (pro: strong signals | con: restrictive / doesn´t work with too many Indicators)__

•	Go Long if all Signals are long

•	Go Short if all Signals are short

•	Go Neutral if Signals are nonunanimous

__Strategy 2 (pro: can be customized | con: more trades / weaker signals)__

•	Go Long if sum of both the Signals > 0 (1+1 / 1+0/ 0+1)

•	Go Short if sum of both the Signals < 0 (-1-1 / -1+0/ 0-1)

•	Go Neutral if sum of both the Signals = 0

# ___Strategy1___
```bash
import SMA_Backtester as SMA
import MeanRev_Backtester as MeanRev
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
from scipy.optimize import minimize
plt.style.use("seaborn")
def optimal_strategy(parameters):
    
    symbol = "price"
    tc = 0.00005
    
    # SMA
    tester1 = SMA.SMABacktester(symbol, int(parameters[0]), int(parameters[1]), tc)
    tester1.test_strategy()
    
    # Bollinger
    tester2 = MeanRev.MeanRevBacktester(symbol,  int(parameters[2]),  int(parameters[3]), tc)
    tester2.test_strategy()
    
    # Create comb
    comb = tester1.results.loc[:, ["returns", "position"]].copy()
    comb.rename(columns = {"position":"position_SMA"}, inplace = True)
    comb["position_MR"] = tester2.results.position
    
    # 2 Methods
    comb["position_comb"] = np.where(comb.position_MR == comb.position_SMA, comb.position_MR, 0) 
    #comb["position_comb"] = np.sign(comb.position_MR + comb.position_SMA)
    
    # Backtest
    comb["strategy"] = comb["position_comb"].shift(1) * comb["returns"]
    comb.dropna(inplace=True)
    comb["trades"] = comb.position_comb.diff().fillna(0).abs()
    comb.strategy = comb.strategy - comb.trades * tc
    comb["creturns"] = comb["returns"].cumsum().apply(np.exp)
    comb["cstrategy"] = comb["strategy"].cumsum().apply(np.exp)
    
    return -comb["cstrategy"].iloc[-1] # negative absolute performance to be minimized
# this function optimises our strategy

bnds =  ((5, 75), (20, 200), (10, 100), (1, 5))
start_par = (5, 20, 10, 1)
opts = minimize(optimal_strategy, start_par, method = "Powell" , bounds = bnds)
opts
# finding the best parameters for Strategy 1

# class for backtesting strategy1

class CombStrategy():
    ''' Class for the vectorized backtesting of SMA-based trading strategies.
    '''
    
    def __init__(self, symbol, SMA_S, SMA_L, SMA, dev, tc):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        '''
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.SMA = SMA
        self.dev = dev
        self.tc = tc
        self.results = None 
        self.get_data()
        
    def __repr__(self):
        return "SMABacktester(symbol = {}, SMA_S = {}, SMA_L = {} )".format(self.symbol, self.SMA_S, self.SMA_L)
        
    def get_data(self):
        ''' Imports the data from BTCUSD.csv (source can be changed).
        '''
        raw = pd.read_csv("BTCUSD_m15.csv", parse_dates = ["date"], index_col = "date", usecols = ['date','bidclose'])
        raw.rename(columns = {'bidclose':'price'}, inplace = True)
        raw = raw[self.symbol].to_frame().dropna()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        raw["SMA_S"] = raw["price"].rolling(self.SMA_S).mean()
        raw["SMA_L"] = raw["price"].rolling(self.SMA_L).mean()
        raw["SMA"] = raw["price"].rolling(self.SMA).mean()
        raw["Lower"] = raw["SMA"] - raw["price"].rolling(self.SMA).std() * self.dev
        raw["Upper"] = raw["SMA"] + raw["price"].rolling(self.SMA).std() * self.dev
        self.data = raw
        return raw
        
    def set_parameters(self, SMA_S = None, SMA_L = None, SMA = None, dev = None):
        ''' Updates SMA parameters and the prepared dataset.
        '''
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["price"].rolling(self.SMA).mean()
            self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
        if dev is not None:
            self.dev = dev
            self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
            
    def test_strategy(self):
        ''' Backtests the SMA-based trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position1"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position1"].shift(1) * data["returns"]
        data["position1"] = data.position1.ffill().fillna(0)
        data["distance"] = data.price - data.SMA
        data["position2"] = np.where(data.price < data.Lower, 1, np.nan)
        data["position2"] = np.where(data.price > data.Upper, -1, data["position2"])
        data["position2"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position2"])
        data["position2"] = data.position2.ffill().fillna(0)
        data["position3"] = np.where(data.position1 == data.position2, data.position1, 0)
        #data["position3"] = np.sign(data.position1 + data.position2)
        data["strategy"] = data.position3.shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine when a trade takes place
        data["trades"] = data.position3.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
       
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | SMA_S = {} | SMA_L = {} | SMA = {} | dev = {} | TC = {}".format(self.symbol, self.SMA_S, self.SMA_L, self.SMA, self.dev, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
test_strategy1.plot_results()
```
test_strategy1.results.trades.value_counts() # only 10 trades taken

As we can see this strategy only took 10 trades, so if you want to avoid overtrading than this strategy is a good option.

# ___Strategy2___
```bash
def optimal_strategy(parameters):
    
    symbol = "price"
    tc = 0.00005
    
    # SMA
    tester1 = SMA.SMABacktester(symbol, int(parameters[0]), int(parameters[1]), tc)
    tester1.test_strategy()
    
    # Bollinger
    tester2 = MeanRev.MeanRevBacktester(symbol,  int(parameters[2]),  int(parameters[3]), tc)
    tester2.test_strategy()
    
    # Create comb
    comb = tester1.results.loc[:, ["returns", "position"]].copy()
    comb.rename(columns = {"position":"position_SMA"}, inplace = True)
    comb["position_MR"] = tester2.results.position
    
    # 2 Methods
    #comb["position_comb"] = np.where(comb.position_MR == comb.position_SMA, comb.position_MR, 0) 
    comb["position_comb"] = np.sign(comb.position_MR + comb.position_SMA)
    
    # Backtest
    comb["strategy"] = comb["position_comb"].shift(1) * comb["returns"]
    comb.dropna(inplace=True)
    comb["trades"] = comb.position_comb.diff().fillna(0).abs()
    comb.strategy = comb.strategy - comb.trades * tc
    comb["creturns"] = comb["returns"].cumsum().apply(np.exp)
    comb["cstrategy"] = comb["strategy"].cumsum().apply(np.exp)
    
    return -comb["cstrategy"].iloc[-1] # negative absolute performance to be minimized

# this function optimises our strategy
bnds =  ((5, 75), (20, 200), (10, 100), (1, 5))
start_par = (5, 20, 10, 1)
opts = minimize(optimal_strategy, start_par, method = "Powell" , bounds = bnds)
opts
# finding the best parameters for Strategy 2
# class for backtesting strategy2

class CombStrategy():
    ''' Class for the vectorized backtesting of SMA-based trading strategies.
    '''
    
    def __init__(self, symbol, SMA_S, SMA_L, SMA, dev, tc):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        '''
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.SMA = SMA
        self.dev = dev
        self.tc = tc
        self.results = None 
        self.get_data()
        
    def __repr__(self):
        return "SMABacktester(symbol = {}, SMA_S = {}, SMA_L = {} )".format(self.symbol, self.SMA_S, self.SMA_L)
        
    def get_data(self):
        ''' Imports the data from BTCUSD.csv (source can be changed).
        '''
        raw = pd.read_csv("BTCUSD_m15.csv", parse_dates = ["date"], index_col = "date", usecols = ['date','bidclose'])
        raw.rename(columns = {'bidclose':'price'}, inplace = True)
        raw = raw[self.symbol].to_frame().dropna()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        raw["SMA_S"] = raw["price"].rolling(self.SMA_S).mean()
        raw["SMA_L"] = raw["price"].rolling(self.SMA_L).mean()
        raw["SMA"] = raw["price"].rolling(self.SMA).mean()
        raw["Lower"] = raw["SMA"] - raw["price"].rolling(self.SMA).std() * self.dev
        raw["Upper"] = raw["SMA"] + raw["price"].rolling(self.SMA).std() * self.dev
        self.data = raw
        return raw
        
    def set_parameters(self, SMA_S = None, SMA_L = None, SMA = None, dev = None):
        ''' Updates SMA parameters and the prepared dataset.
        '''
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["price"].rolling(self.SMA).mean()
            self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
        if dev is not None:
            self.dev = dev
            self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
            
    def test_strategy(self):
        ''' Backtests the SMA-based trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position1"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position1"].shift(1) * data["returns"]
        data["position1"] = data.position1.ffill().fillna(0)
        data["distance"] = data.price - data.SMA
        data["position2"] = np.where(data.price < data.Lower, 1, np.nan)
        data["position2"] = np.where(data.price > data.Upper, -1, data["position2"])
        data["position2"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position2"])
        data["position2"] = data.position2.ffill().fillna(0)
        #data["position3"] = np.where(data.position1 == data.position2, data.position1, 0)
        data["position3"] = np.sign(data.position1 + data.position2)
        data["strategy"] = data.position3.shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine when a trade takes place
        data["trades"] = data.position3.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
       
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | SMA_S = {} | SMA_L = {} | SMA = {} | dev = {} | TC = {}".format(self.symbol, self.SMA_S, self.SMA_L, self.SMA, self.dev, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
test_strategy2.plot_results()
```
test_strategy2.results.trades.value_counts() # 56 long, 279 short positions taken by this strategy

As we can see this is a much better strategy with higher returns than previous one

## _Implementing the strategy in FXCM_
```bash
class Trader():
    
    def __init__(self, instrument, bar_length, SMA, dev, SMA_S, SMA_L, units):
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length) 
        self.tick_data = None
        self.raw_data = None
        self.data = None 
        self.ticks = 0
        self.last_bar = None 
        self.units = units
        self.position = 0
        
        #*****************add strategy-specific attributes here******************
        self.SMA = SMA
        self.dev = dev
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        #************************************************************************        
    
    def get_most_recent(self, period = "m1", number = 10000):
        while True:  
            time.sleep(5)
            df = api.get_candles(self.instrument, number = number, period = period, columns = ["bidclose", "askclose"])
            df[self.instrument] = (df.bidclose + df.askclose) / 2
            df = df[self.instrument].to_frame()
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()) - self.last_bar < self.bar_length:
                break
    
    def get_tick_data(self, data, dataframe):
        
        self.ticks += 1
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(data["Updated"], unit = "ms")
        
        if recent_tick - self.last_bar > self.bar_length:
            self.tick_data = dataframe.loc[self.last_bar:, ["Bid", "Ask"]]
            self.tick_data[self.instrument] = (self.tick_data.Ask + self.tick_data.Bid)/2
            self.tick_data = self.tick_data[self.instrument].to_frame()
            self.resample_and_join()
            self.define_strategy() 
            self.execute_trades()
            
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                             label="right").last().ffill().iloc[:-1])
        self.last_bar = self.raw_data.index[-1]  
        
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        df = df.append(self.tick_data.iloc[-1]) # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["SMA_S"] = df[self.instrument].rolling(self.SMA_S).mean()
        df["SMA_L"] = df[self.instrument].rolling(self.SMA_L).mean()
        df["SMA"] = df[self.instrument].rolling(self.SMA).mean()
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev
        df["distance"] = df.self.instrument - df.SMA
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev
        df.dropna(inplace = True)
        
        
        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position_SMA"] = np.where(df["SMA_S"] > df["SMA_L"], 1, -1 )
        df["position_BB"] = np.where(df.self.instrument < df.Lower, 1, np.nan)
        df["position_BB"] = np.where(df.self.instrument > df.Upper, -1, df["position_BB"])
        df["position_BB"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["position_BB"])

        df["position_comb"] = np.sign(df.position_MR + df.position_SMA)
        df["position_comb"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position_comb"].iloc[-1] == 1:
            if self.position == 0:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING LONG")  
            self.position = 1
        elif self.data["position_comb"].iloc[-1] == -1: 
            if self.position == 0:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING SHORT")  
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING SHORT")  
            self.position = -1
        elif self.data["position_comb"].iloc[-1] == 0: 
            if self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL") 
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL")  
            self.position = 0

    def report_trade(self, order, going):
        time = order.get_time()
        units = api.get_open_positions().amountK.iloc[-1]
        price = api.get_open_positions().open.iloc[-1]
        unreal_pl = api.get_open_positions().grossPL.sum()
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | Unreal. P&L = {}".format(time, units, price, unreal_pl))
        print(100 * "-" + "\n")
trader = Trader("BTC/USD", bar_length = "5min",SMA = 31, dev = 2.52, SMA_S = 53, SMA_L = 109, units = 100)
trader.get_most_recent()
api.subscribe_market_data(trader.instrument, (trader.get_tick_data, ))
api.unsubscribe_market_data(trader.instrument)
if len(api.get_open_positions()) != 0: # if we have final open position(s) (netting and hedging)
    api.close_all_for_symbol(trader.instrument)
    print(2*"\n" + "{} | GOING NEUTRAL".format(str(datetime.utcnow())) + "\n")
    time.sleep(20)
    print(api.get_closed_positions_summary()[col])
    trader.position = 0
trader.data
api.close()
```
### Now I am deploying this strategy in AWS. To create it truly automated, I am creating the automated trading bot with the below code (present as ‘Automated Bot.py’ file) – 
```bash
import pandas as pd
import numpy as np
import fxcmpy
from datetime import datetime, timedelta
import time


col = ["tradeId", "amountK", "currency", "grossPL", "isBuy"]


class Trader():
    
    def __init__(self, instrument, bar_length, SMA, dev, SMA_S, SMA_L, units):
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length) 
        self.tick_data = None
        self.raw_data = None
        self.data = None 
        self.ticks = 0
        self.last_bar = None 
        self.units = units
        self.position = 0
        
        #*****************add strategy-specific attributes here******************
        self.SMA = SMA
        self.dev = dev
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        #************************************************************************        
    
    def get_most_recent(self, period = "m1", number = 10000):
        while True:  
            time.sleep(5)
            df = api.get_candles(self.instrument, number = number, period = period, columns = ["bidclose", "askclose"])
            df[self.instrument] = (df.bidclose + df.askclose) / 2
            df = df[self.instrument].to_frame()
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()) - self.last_bar < self.bar_length:
                break
    
    def get_tick_data(self, data, dataframe):
        
        self.ticks += 1
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(data["Updated"], unit = "ms")
        
        # define stop(to stop live stream)
        if recent_tick.time() >= pd.to_datetime("20:30").time():
            print("Stop the Trading Session!")
            api.unsubscribe_market_data(self.instrument)
            if len(api.get_open_positions()) != 0:
                api.close_all_for_symbol(self.instrument)
                print(2*"\n" + "{} | GOING NEUTRAL".format(str(datetime.utcnow())) + "\n")
                time.sleep(20)
                print(api.get_closed_positions_summary()[col])
                self.position = 0
                api.close()
        
        if recent_tick - self.last_bar > self.bar_length:
            self.tick_data = dataframe.loc[self.last_bar:, ["Bid", "Ask"]]
            self.tick_data[self.instrument] = (self.tick_data.Ask + self.tick_data.Bid)/2
            self.tick_data = self.tick_data[self.instrument].to_frame()
            self.resample_and_join()
            self.define_strategy() 
            self.execute_trades()
            
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                             label="right").last().ffill().iloc[:-1])
        self.last_bar = self.raw_data.index[-1]  
        
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        df = df.append(self.tick_data.iloc[-1]) # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["SMA_S"] = df[self.instrument].rolling(self.SMA_S).mean()
        df["SMA_L"] = df[self.instrument].rolling(self.SMA_L).mean()
        df["SMA"] = df[self.instrument].rolling(self.SMA).mean()
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev
        df["distance"] = df.self.instrument - df.SMA
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev
        df.dropna(inplace = True)
        
        
        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position_SMA"] = np.where(df["SMA_S"] > df["SMA_L"], 1, -1 )
        df["position_BB"] = np.where(df.self.instrument < df.Lower, 1, np.nan)
        df["position_BB"] = np.where(df.self.instrument > df.Upper, -1, df["position_BB"])
        df["position_BB"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["position_BB"])

        df["position_comb"] = np.sign(df.position_MR + df.position_SMA)
        df["position_comb"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position_comb"].iloc[-1] == 1:
            if self.position == 0:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING LONG")  
            self.position = 1
        elif self.data["position_comb"].iloc[-1] == -1: 
            if self.position == 0:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING SHORT")  
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING SHORT")  
            self.position = -1
        elif self.data["position_comb"].iloc[-1] == 0: 
            if self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL") 
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL")  
            self.position = 0

    def report_trade(self, order, going):
        time = order.get_time()
        units = api.get_open_positions().amountK.iloc[-1]
        price = api.get_open_positions().open.iloc[-1]
        unreal_pl = api.get_open_positions().grossPL.sum()
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | Unreal. P&L = {}".format(time, units, price, unreal_pl))
        print(100 * "-" + "\n")


if __name__ == "__main__":  
    api = fxcmpy.fxcmpy(config_file = r"C:\Users\studd\Downloads\JOB\projects\AI trading bot\AWS Automated  Model\FXCM.cfg")
    trader = Trader("BTC/USD", bar_length = "15min",SMA = 31, dev = 2.52, SMA_S = 53, SMA_L = 109, units = 100)
    trader.get_most_recent()
    api.subscribe_market_data(trader.instrument, (trader.get_tick_data, ))
```

# AWS Automation complete process:

_Step 1: Logged in to my AWS account(aws link)_

_Step 2: Created an EC2 instance(t2 micro is used under free tier)_

_Step 3: Create a key pair(.pem file) of your instance_

_Step 4: Connect to the instance created(go to instance->connect->download remote desktop file->get password->use the .pem file->get the password->log in from the remote desktop instance)_

_Step 5: Download and install Anaconda and required packages in remote desktop instance_

_Step 6: Enable usage of local drive(which contains the codes) for the remote desktop instance_

_Step 7: Copy paste the files needed in the instance_

_Step 8: Create a .bat file(notepad file with automated trading file path). Then on double clicking the .bat file, the command prompt opens automatically and the code file Automated Bot.py runs_

_Step 9: Now finally to automate entire process we schedule the process with task schedular and schedule the week days and start time to start trading._

_Step 10: We set the stop time in the Trader class of Automated Bot.py file._

## ___By this the completely automated trading bot is created!___

```bash
```
```bash
```