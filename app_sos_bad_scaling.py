#Packages Import
import math # Mathematical functions 
import numpy as np 
import pandas as pd 
from datetime import date, timedelta, datetime 
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates # Formatting dates
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
import seaborn as sns

scaler_pred = MinMaxScaler()
# Data retrieval for cryptocurrencies BTCUSDT ETHUSDT XMRUSDT 
# Binance API credentials 
apikey = '2ZaEiNukiFtPHqnYdCGMANNimHdcCF0nvv4L9eYXMIdr4ovveQrca4NWKsZ7DxAy'
secretkey = '9zjousoyG0t2wblcGkm0tVGLBfz6Woo3paakGh49Vhj0IxrGGXl332b1KQaxlJKY'

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
#API Initialization 
client = Client(apikey, secretkey)

#Actual Prediction

#Packages to load previously scaled data in array format for each crypto prediction in the below if statements
# load numpy array from npy file
import keras
from numpy import load

import pickle 

df = pd.read_pickle('DataSets\BTCUSDT_3MIN')
#Checking that it is a DF before moving one
type(df)


df_train = df.sort_values(by=['Open Time']).copy()

# Saving a copy of the dates index, before we need to reset it to numbers
date_index = df_train.index

# We reset the index, so we can convert the date-index to a number-index
# df_train = df_train.reset_index(drop=True).copy()
df_train.drop(columns=['index'], inplace=True, axis = 1)

def prepare_data(df):

    # List of considered Features
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

    print('FEATURE LIST')
    print([f for f in FEATURES])

    # Create the dataset with features and filter the data to the list of FEATURES
    df_filter = df[FEATURES]
    
    # Convert the data to numpy values
    np_filter_unscaled = np.array(df_filter)
    #np_filter_unscaled = np.reshape(np_unscaled, (df_filter.shape[0], -1))
    print(np_filter_unscaled.shape)

    np_c_unscaled = np.array(df['Close']).reshape(-1, 1)
    
    return np_filter_unscaled, np_c_unscaled
    
np_filter_unscaled, np_c_unscaled = prepare_data(df_train)
                                          
# Creating a separate scaler that works on a single column for scaling predictions
# Scale each feature to a range between 0 and 1

scaler_train = MinMaxScaler()
np_scaled = scaler_train.fit_transform(np_filter_unscaled)
    
# Create a separate scaler for a single column
scaler_pred = MinMaxScaler()
scaler_pred.fit_transform(np_c_unscaled)   


def prediction(crypto):
  #Choose between BTCUSDT ETHUSDT XMRUSDT
  if crypto == "BTCUSDT":
    #Fetch Last Day 3min Candles Data
    klines = client.get_historical_klines(crypto, Client.KLINE_INTERVAL_3MINUTE, "1 day ago UTC")

    pred_df = pd.DataFrame(klines)
    pred_df.columns = (['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume','Nb of Trade', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume','Ignored'])
    # hist_df.drop(labels = ['TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignored', 'Quote Asset Volume'], inplace = True,axis = 1)
    pred_df['Close Time'] = pd.to_datetime(pred_df['Close Time']/1000, unit='s')
    # hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')
    pred_df.drop(['Ignored', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Quote Asset Volume'], inplace= True, axis = 1)

    #load scaled xmr array
    np_scaled = load('scaled/btc_np_scaled.npy')

    pred_df = pred_df.apply(pd.to_numeric)
    pred = np_scaled[-51:-1,:].reshape(1,50,5)

    # pred_df.to_csv('5H_avg_pred_df')
    # pred_df
    model = keras.models.load_model('Models\BTC_Model_3MIN.h5')
    btc_y_pred_scaled = model(pred)
    btc_y_pred = scaler_pred.inverse_transform(btc_y_pred_scaled) 
    return btc_y_pred
  elif crypto == "XMRUSDT":

    #Fetch Last Day 3MINUTE Candles Data
    klines = client.get_historical_klines(crypto, Client.KLINE_INTERVAL_3MINUTE, "1 day ago UTC")

    pred_df = pd.DataFrame(klines)
    pred_df.columns = (['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume','Nb of Trade', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume','Ignored'])
    # hist_df.drop(labels = ['TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignored', 'Quote Asset Volume'], inplace = True,axis = 1)
    pred_df['Close Time'] = pd.to_datetime(pred_df['Close Time']/1000, unit='s')
    # hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')
    pred_df.drop(['Ignored', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Quote Asset Volume'], inplace= True, axis = 1)

    #load scaled xmr array
    np_scaled = load(r'scaled\xmr_np_scaled.npy')

    xmr_pred_df = pred_df.apply(pd.to_numeric)
    xmr_pred = np_scaled[-51:-1,:].reshape(1,50,5)

    # pred_df.to_csv('5H_avg_pred_df')
    # pred_df
    xmr_model = keras.models.load_model('Models\XMR_Model_3MIN.h5')
    xmr_y_pred_scaled = xmr_model(xmr_pred)
    xmr_y_pred = scaler_pred.inverse_transform(xmr_y_pred_scaled)

    return xmr_y_pred
      
  elif crypto == "ETHUSDT":
    #Fetch Last Day  3min Candles Data
    klines = client.get_historical_klines(crypto, Client.KLINE_INTERVAL_3MINUTE, "1 day ago UTC")

    pred_df = pd.DataFrame(klines)
    pred_df.columns = (['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume','Nb of Trade', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume','Ignored'])
    # hist_df.drop(labels = ['TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignored', 'Quote Asset Volume'], inplace = True,axis = 1)
    pred_df['Close Time'] = pd.to_datetime(pred_df['Close Time']/1000, unit='s')
    # hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')
    pred_df.drop(['Ignored', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Quote Asset Volume'], inplace= True, axis = 1)
    # hist_df['Open Time'] = hist_df.index

    #load scaled xmr array
    np_scaled = load(r'scaled/eth_np_scaled.npy')
    eth_pred_df = pred_df.apply(pd.to_numeric)
    eth_pred = np_scaled[-51:-1,:].reshape(1,50,5)

    eth_model = keras.models.load_model('Models\ETH_Model_3MIN.h5')
    eth_y_pred_scaled = eth_model(eth_pred)
    eth_y_pred = scaler_pred.inverse_transform(eth_y_pred_scaled)

    return eth_y_pred

if __name__ == '__main__':
   app.run()
