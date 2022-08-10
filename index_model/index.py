import datetime as dt
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import numpy as np
import pandas as pd
import os
current_dir = os.getcwd()


#%% Class for contructing the Index and generating all the stats for a stock
class IndexModel():
    '''Class to construct and analyse the Index
    
    Attributes
    ==========

    start: str
        start date of the data
    end: str
        end date of the data 
        
    Index Computation
    =================
    
    weights: list
        weights of the assets composing the index        
        
    '''
    def __init__(self, start, end, weights = [0.5, 0.25, 0.25]):
        self.start = start
        self.end = end
        self.weights = weights
        self.get_data()
     
    def __repr__(self): #repr stands for representation. Add comments that the user should know whenyou run this class
        return "IndexModel(start = {}, end = {})".format( self.start, self.end)   
    
    def get_data(self):
        '''Retrives data from .csv file
        '''
        total_price = pd.read_csv(current_dir + "\data_sources\stock_prices.csv", index_col = [0])
        # Making sure the prices are base 100
        total_price = total_price.div(total_price.iloc[0]).mul(100)
        # Replacing NaN with the last price
        total_price = total_price.fillna(method = "ffill") #bfill, .interpolate()
        
        self.stock_data = total_price
        
    def calc_index_level(self):
        total_price = self.stock_data
        start = self.start
        end = self.end
        weights = self.weights
        
        if len(weights) > len(list(total_price)):
            raise ValueError('More weights provided than assets in the investable universe')
        if sum(weights) != 1:
            raise ValueError('Weights do not add to 100%')
            
        return total_price

    def export_values(self, file_name: str) -> None:
        # To be implemented
        pass
               
    def plot_prices(self):
        self.stock_data.price.plot(figsize=(12,8))
        plt.title("Price Chart: {}".format(self._ticker), fontsize = 15)
        
    def plot_returns(self, kind = "ts"):
        ''' Plots log returns either as time series ("ts") or as histogram ("hist")
        '''
        if kind == "ts":            
            self.stock_data.log_returns.plot(figsize=(12,8))
            plt.title("Returns:{}".format(self._ticker), fontsize = 15)
        elif kind == "hist":
            self.stock_data.log_returns.hist(figsize=(12,8), bins = int(np.sqrt(len(self.stock_data))))
            plt.title("Frequency of Returns:{}".format(self._ticker), fontsize = 15)            
           
class RiskReturn(IndexModel):

    # When we pass __init__(self, ticker, start, end), the child class RiskReturn overrides the parent class and since 
    # RiskReturn does not download data from yf, this will result in error, hence we need the super function
       
    def __init__(self, start, end, freq = None):
        self.freq = freq
        super().__init__(start, end) 
    def __repr__(self): #repr stands for representation. Add comments that the user should know whenyou run this class
        return "RiskReturn(ticker = {}, start = {}, end = {})".format(self._ticker, self.start, self.end)   
    def mean_return(self):
        ''' Calculates mean return
        '''
        if self.freq is None:
            return self.stock_data.log_returns.mean()
        else:
            resampled_price = self.stock_data.price.resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()   
    def std_returns(self):
        ''' Calculates the standard deviation of returns (risk)
        '''
        if self.freq is None:
            return self.stock_data.log_returns.std()
        else:
            resampled_price = self.stock_data.price.resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()        
    def annualized_perf(self):
        ''' Calculates annulized return and risk
        '''
        mean_return = round(self.stock_data.log_returns.mean() * 252, 3)
        risk = round(self.stock_data.log_returns.std() * np.sqrt(252), 3)
        print("Return: {} | Risk: {}".format(mean_return, risk))