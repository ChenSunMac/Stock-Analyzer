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

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
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
    
    
def visualize_data():
    # Read Data From the Main Data Set
    df = pd.read_csv('sp500_joined_closes.csv')
    # create dataFrame using correlation()
    df_corr = df.corr()
    print(df_corr.head())
    # Save the correlation result in .csv format
    df_corr.to_csv('sp500corr.csv')
    # List of values from DataFrame    
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)    
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)
    # set x and y axis ticks
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()

    
#visualize_data()


"""
Pre-Process Data for Machine Learning
    Find Features and Labels from SP500 data:
        Features:
        Labels: 
            - BUY : within the next 7 days, its price goes up more than 2%
            - SELL : within the next 10 days, its price goes down more than 2%
            - HOLD: meh~
"""
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace = True)
    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]    
    df.fillna(0, inplace=True)
    return tickers, df  

tickers, df = process_data_for_labels('AMD')


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement: 
            return 1
        if col < -requirement:
            return -1
        return 0
    
from collections import Counter
def extract_featuresets(ticker):
    """
    @ param:
    @ return: 
        X: feature sets values (daily % changes for every company in the S&P 500)
        Y: label 
    """
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)    
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)   

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X,y,df

X, y, df = extract_featuresets('AMD')


from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                        y,
                                                        test_size=0.25)
    # Using KNN Classifier
    # clf = neighbors.KNeighborsClassifier()
    # Using Voting Classifier instead of normal KNN
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])    
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    prediction = clf.predict(X_test)
    print('predicted class counts: ', Counter(prediction))
    
    
do_ml('AMD')    











    
    
    
    
    
    
    