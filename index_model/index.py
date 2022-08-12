import matplotlib.pyplot as plt
from scipy.stats import rankdata
plt.style.use("seaborn")
import numpy as np
import pandas as pd
import os
current_dir = os.getcwd()


#%% Class for contructing the Index and generating all the stats for a stock
class IndexModel():
    """
    Class to construct and analyse the Index
    
    Attributes
    ----------

    start: str
        start date of the data
    end: str
        end date of the data 
        
    Index Computation
    -----------------
    
    weights: list
        weights of the assets composing the index        
        
    """
    def __init__(self, start, end, weights = [0.5, 0.25, 0.25]):
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.weights = weights
        self.get_data()
     
    def __repr__(self): #repr stands for representation. Add comments that the user should know whenyou run this class
        return "IndexModel(start = {}, end = {})".format( self.start, self.end)   
    
    def get_data(self):
        '''Retrives data from .csv file
        '''
        total_price = pd.read_csv(current_dir + "\data_sources\stock_prices.csv", index_col = [0])
        # test = pd.read_csv(current_dir + "\data_sources\index_level_results_rounded.csv", index_col = [0])
        total_price.index = pd.to_datetime(total_price.index, format="%d/%m/%Y")
        # Making sure the prices are base 100
        total_price = total_price.div(total_price.iloc[0]).mul(100)
        # Replacing NaN with the last price
        total_price = total_price.fillna(method = "bfill")
        
        self.total_price = total_price
        
    def calc_index_level(self):
        """
        Assumptions:
        -----------
            Each asset has equal stocks outstanding, hence the stock price is the Market Cap of each stock 
            Weights are assigned in the Descending order of the Market Cap
            Weights are provided by the user
            
        Raises
        ------
        ValueError
            Checks on acceptable number of assets, sum of weights .
        Exception
            Checks on start-date and end-date.

        Returns
        -------
        index_price : Pandas DataFrame
            The Pandas DataFrame of the Price series of the Index

        """
        total_price = self.total_price
        start = self.start
        end = self.end
        weights = self.weights 
        
        # Retriving the daily dates series
        daily_date_series = self.total_price.index     
        
        # Retriving the monthly dates series of the first business day in each month
        monthly_date_series = self.total_price.resample("BMS").last().index
        
        # Adding the first and the last day
        monthly_date_series = daily_date_series[0:1].union(monthly_date_series).union(daily_date_series[-1:])
        
        # Raising all the necessary exceptions
        if len(weights) > len(list(total_price)):
            raise ValueError("More weights provided than assets in the investable universe")
        if sum(weights) != 1:
            raise ValueError("Weights do not add to 100%")
        if start < monthly_date_series[0]:
            raise Exception("Start-date preceeds the starting date of the data available. Start-date reset to the {}".format(monthly_date_series[1]))
            start = monthly_date_series[1]
            self.start = start
        if end > monthly_date_series[-1]:
            raise Exception("End-date exceeds the data available. End-date reset to the {}".format(monthly_date_series[-1]))
            end = monthly_date_series[-1] 
            self.end = end
        
        # Composing the Cap-Weighted Index
        # Finding the address of the starting date (or the closest starting date)
        start_date_address = monthly_date_series.get_indexer([start], method='nearest')[0]
        end_date_address = monthly_date_series.get_indexer([end], method='nearest')[0]
        rebalancing_dates = monthly_date_series[start_date_address:(end_date_address+1)]
        
        # Constructing the Index
        index_price = index_modeller(rebalancing_dates, total_price, weights)
        
        return index_price

    def export_values(self, file_name: str) -> None:
        # To be implemented
        pass
               
    def plot_prices(self):
        self.total_price.price.plot(figsize=(12,8))
        plt.title("Price Chart: {}".format(self._ticker), fontsize = 15)
        
    def plot_returns(self, kind = "ts"):
        ''' Plots log returns either as time series ("ts") or as histogram ("hist")
        '''
        if kind == "ts":            
            self.total_price.log_returns.plot(figsize=(12,8))
            plt.title("Returns:{}".format(self._ticker), fontsize = 15)
        elif kind == "hist":
            self.total_price.log_returns.hist(figsize=(12,8), bins = int(np.sqrt(len(self.total_price))))
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
            return self.total_price.log_returns.mean()
        else:
            resampled_price = self.total_price.price.resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()   
    def std_returns(self):
        ''' Calculates the standard deviation of returns (risk)
        '''
        if self.freq is None:
            return self.total_price.log_returns.std()
        else:
            resampled_price = self.total_price.price.resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()        
    def annualized_perf(self):
        ''' Calculates annulized return and risk
        '''
        mean_return = round(self.total_price.log_returns.mean() * 252, 3)
        risk = round(self.total_price.log_returns.std() * np.sqrt(252), 3)
        print("Return: {} | Risk: {}".format(mean_return, risk))

def index_modeller(rebalancing_dates, total_price, weights):
    """
    Assumptions:
    -----------
        Each asset has equal stocks outstanding, hence the stock price is the Market Cap of each stock 
        Weights are assigned in the Descending order of the Market Cap
        Weights are provided by the user

    Parameters
    ----------
    rebalancing_dates : list
        The days on which we are going to perform the rebalancing
    total_price : DataFrame
        Total Return Price of the assets composing the Index
    weights : list
        Weights allocated to the assets in the Index

    Returns
    -------
    The Pandas DataFrame of the Price series of the Index

    """
    list_of_stocks = list(total_price.copy())
    total_num_of_stocks = list(range(len(list_of_stocks)))
    chosen_num_of_stocks = len(weights)
    ranked_assets = total_num_of_stocks[-chosen_num_of_stocks:]
    initial_price = 100
    
    # Adding needed drifting assets
    drifting_assets = []
    for ii in range(chosen_num_of_stocks):
        drifting_assets.append("Asset"+str(ii+1))
    
    # Parsing the total_price DataFrame over the required period
    total_price = total_price.loc[rebalancing_dates[0]:rebalancing_dates[-1]]
    
    # Initializing the price series for each asset
    total_price[drifting_assets] = weights*np.array(initial_price)
    total_price["Index_Level"] = initial_price
    
    horizon = np.shape(total_price)[0]
    dates = total_price.index
    
    # Initializing the rebalancings
    rebalancing_counts = 0
    
    for i in range(horizon):      

        if dates[i] in rebalancing_dates:
            # For the current rebalancing lets first generate the order in which to apply our weights
            address_ranked_assets = []
            current_ranks = rankdata(total_price.loc[rebalancing_dates[rebalancing_counts],list_of_stocks].values)
            for j in range(chosen_num_of_stocks):
                address_ranked_assets.append(np.where(current_ranks == (ranked_assets[j]+1))[0][0])
            # The order in which to apply the weights
            address_ranked_assets.reverse()
            rebalancing_counts = rebalancing_counts + 1
            is_it_rebalancing = "y"
        else:
            is_it_rebalancing = "n"
        
        if i > 0:
            if is_it_rebalancing == "y":
                total_price.loc[dates[i],drifting_assets] = total_price.loc[dates[i-1],drifting_assets]*weights
            elif is_it_rebalancing == "n":
                for jj in range(chosen_num_of_stocks):
                    total_price.loc[dates[i],drifting_assets[jj]] = total_price.loc[dates[i-1],drifting_assets[jj]] * (total_price.loc[dates[i],list_of_stocks[address_ranked_assets[jj]]]/total_price.loc[dates[i-1],list_of_stocks[address_ranked_assets[jj]]])
                total_price.loc[dates[i],"Index_Level"] =sum(total_price.loc[dates[i],drifting_assets])                 
                    
            
            