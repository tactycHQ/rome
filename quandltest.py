import quandl as qdl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load API Key
qdl.ApiConfig.api_key = "PW_S2NKKWSoUccp-RKvX"

#Creation of Quandl API Call
myTicker = ['AAPL']
start_date = '2016-01-01'
end_date = '2018-12-31'
columns = ['ticker','date','close','volume']

qdl_data = qdl.get_table('WIKI/PRICES',
                     qopts = {'columns':columns},
                     ticker = myTicker,
                     date={'gte':start_date,'lte':end_date})

#Slicing prices and dates
prices = qdl_data.loc[:,'close'].values
dates = qdl_data.loc[:,'date'].values


#Printing stock price chart
print(qdl_data.head(5))
plt.figure(1)
plt.plot(dates, prices, marker='o')
plt.xlabel('Date')
plt.ylabel('Stock Price')
# plt.show()