# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:47:15 2017

@author: chens
"""

import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import fix_yahoo_finance 
fix_yahoo_finance.pdr_override()

import time



def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "html5lib")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:101]:
        ticker = row.findAll('td')[0].text
        if '.' in ticker:
            continue
        else:
            tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers


tickers = save_sp500_tickers()
print(tickers)

def get_data_from_yahoo(reload_sp500=False):   
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2017, 11, 19)
    
    for ticker in tickers[:]:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.get_data_yahoo(ticker, start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
            time.sleep(2)
        else:
            print('Already have {}'.format(ticker))

"""
NEED FURTHER DEVELOPMENT
def update_data_from_yahoo(reload_sp500 = False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
            
    for ticker in tickers[:]:
        df_time_row = pd.read_csv('stock_dfs/{}.csv'.format(ticker), 
                                parse_dates = True, index_col = 0).tail()
        start_time = df_time_row['Date'][df_time_row.last_valid_index()]
        year, month, date = start_time.split()
        start = dt.datetime(int(year), int(month), int(date))
"""        

def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
    main_df = pd.DataFrame()    
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)        

        df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']
        
        df.rename(columns={'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how = 'outer')
            
        if count % 10 == 0:
            print(count)            
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')
#get_data_from_yahoo()   
#compile_data()
    
    
    
    
    
    
    
    
    
    