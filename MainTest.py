# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:20:48 2017

@author: chens
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2005,1,1)
end = dt.datetime(2017,11,20)
"""
READ DATA from online API:
    using pandas_datareader get company stock data from yahoo
    convert data to .csv form by .tocsv()
"""
df = web.DataReader('AMD', 'yahoo', start, end)
df.to_csv('amd.csv')
# read from csv with dates parsed as index:
df = pd.read_csv('amd.csv', parse_dates= True, index_col = 0)

# just plot the 'Adj Close' col
#df['Adj Close'].plot()
#plt.show()

"""
 SMOOTHING THE DATA:
     100 moving average for smoothing the stock data
     - dropna() to drop off the rows with NA result( can be avoided if we set min_periods = 0)
"""
df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()
df.dropna(inplace = True)
print(df.head())

"""
 (OHLC)Open-High-Low-Close chart:
     - Use candlestick_ohlc to plot ohlc chart
     - shrink the data using resample 
"""
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
df_ohlc.reset_index(inplace = True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex = ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1, df_ohlc.values, width = 2, colorup = 'g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values, 0)
plt.show()




#ax1.plot(df.index, df['Adj Close'])
#ax1.plot(df.index, df['100ma'])
#ax2.bar(df.index, df['Volume'])

