from iexfinance import get_historical_data
from datetime import datetime
import pandas as pd
import numpy as np


def main():
    # aapl_data=getData("AAPL",)
    # saveData(aapl_data)
    raw_data=loadData('out.csv')
    features,targets,dates=PrepData(raw_data)
    createInputs(features,targets)

def getData(ticker):
    start=datetime(2013,2,9)
    end = datetime(2017, 5, 9)
    stock_data = get_historical_data(ticker,start=start,end=end,output_format='pandas')
    print(stock_data.describe)
    return stock_data

def saveData(dataframe):
    dataframe.to_csv('out.csv')
    print("Saved")

def loadData(filename):
    raw_data = pd.read_csv(filename)
    print("------Loaded Raw Data-------")
    return raw_data

def PrepData(df):

    feature_set=['close','volume']
    targets=df['close'].values.reshape(-1,1)
    dates=df['date'].values.reshape(-1,1)
    features=pd.DataFrame(df,columns=feature_set).values

    print("Feature Shape is ",features.shape)
    print("Target Shape is ", targets.shape)
    print("Dates Shape is ",dates.shape)

    features_mean=features.mean(axis=0)


    return features,targets,dates

def createInputs(features,targets,train_ratio=0.8):

    train_index=round(len(features)*train_ratio)
    x_train = features[:train_index,:]
    y_train = targets[:train_index,:]
    x_test = features[train_index:, :]
    y_test = targets[train_index:, :]

    print("-----Creating Model Inputs-----")
    print("x_train Shape is ", x_train.shape)
    print("y_train Shape is ", y_train.shape)
    print("x_test Shape is ", x_test.shape)
    print("y_test Shape is ", y_test.shape)

    return x_train,y_train,x_test,y_test

def build_model(x_train,y_train):








if __name__ == '__main__':
    main()
