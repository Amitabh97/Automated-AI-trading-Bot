
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