

from iexfinance import get_historical_data
import pandas as pd
import numpy as np
import datetime as dt


class DataLoader():

    def __init__(self,ticker):
        self.ticker=ticker
        self.stock_data=None
        self.fname = self.ticker + '.csv'
        self.features=None
        self.targets=None
        self.dates=None
        self.features_mean=None
        self.features_std=None
        self.targets_mean = None
        self.targets_std = None
        self.feature_set = ['close', 'volume']

    def getData(self):
        start = dt.datetime(2013, 2, 9)
        end = dt.datetime(2017, 5, 9)
        self.stock_data = get_historical_data(self.ticker, start=start, end=end, output_format='pandas')
        self.stock_data.to_csv(self.fname)
        print("Saved to", self.fname)

        return self.stock_data

    def loadData(self):
        self.stock_data = pd.read_csv(self.fname)
        print("------Loaded Raw Data-------")

    def PrepData(self):

        self.targets = self.stock_data['close'].values.reshape(-1, 1)
        self.dates = self.stock_data['date'].values.reshape(-1, 1)
        self.features = pd.DataFrame(self.stock_data, columns=self.feature_set).values

        self.targets, self.targets_mean, self.targets_std = self.normalize(self.targets)

        print("Feature Shape is ", self.features.shape)
        print("Target Shape is ", self.targets.shape)
        print("Dates Shape is ", self.dates.shape)


    def normalize(self,denorm):
        mean = denorm.mean(axis=0)
        std = denorm.std(axis=0)
        norm = (denorm - mean) / std
        return norm, mean, std

    def denormalize(self,norm,mean,std):
        denorm = norm * std + mean





