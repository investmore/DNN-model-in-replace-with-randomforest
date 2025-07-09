import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import numpy as np
class tech_indicators:
    def __init__(self,data,stock):
        self.data = data
        self.stock = stock
    def stoch_R(self, n, m):
        data = self.data[self.stock]
        storeddata = pd.DataFrame(index=data.index)
        storeddata['lowest_l14'] = data['Low'].rolling(window=n).min()
        storeddata['highest_h14'] = data['High'].rolling(window=n).max()
        storeddata['%K'] = (data['Close'] - storeddata['lowest_l14']) / (
            storeddata['highest_h14'] - storeddata['lowest_l14']
        ) * 100
        storeddata['%K'].fillna(0, inplace=True)
        storeddata['%D'] = storeddata['%K'].rolling(window=m).mean()
        return storeddata['%D'][n:]
    def kalmanfilter(self,n):
        data = self.data[self.stock]
        kf = KalmanFilter(transition_matrices = [1],
         observation_matrices = [1],
         initial_state_mean = 0,
         initial_state_covariance = 1,
         observation_covariance=1,
         transition_covariance=.0001)
        mean, cov = kf.filter(data['Close'].values)
        mean = pd.Series(mean.flatten(), index=data.index)
        ratio = mean[n:]/data['Close'][n:]
        return ratio
    def ADX(self,n):
        data = self.data[self.stock]
        period = n
        data['TR'] = np.maximum(data['High'] - data['Low'],
                            np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                       abs(data['Low'] - data['Close'].shift(1))))
        data['+DM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
                           np.maximum(data['High'] - data['High'].shift(1), 0), 0)
        data['-DM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
                           np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)
        data['TR_smooth'] = data['TR'].rolling(window=period).mean()
        data['+DM_smooth'] = data['+DM'].rolling(window=period).mean()
        data['-DM_smooth'] = data['-DM'].rolling(window=period).mean()
        data['+DI'] = (data['+DM_smooth'] / data['TR_smooth']) * 100
        data['-DI'] = (data['-DM_smooth'] / data['TR_smooth']) * 100
        data['DX'] = (abs(data['+DI'] - data['-DI']) / abs(data['+DI'] + data['-DI'])) * 100
        data['ADX'] = data['DX'].rolling(window=period).mean()
        data['signal'] = (data['+DI'] > data['-DI']).astype(int)
        return data['ADX'][n:],data['signal'][n:]
    def ATR(self,n):
        data = self.data[self.stock]
        period = n
        data['High-Low'] = data['High'] - data['Low']
        data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
        data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
        data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        data['ATR'] = data['TR'].rolling(window=period).mean()
        data['Upper_Break'] = data['Close'] > (data['Close'].shift(1) + data['ATR'])
        data['Lower_Break'] = data['Close'] < (data['Close'].shift(1) - data['ATR'])
        data['Buy_Signal'] = data['Upper_Break'].astype(int)   # Buy signal = 1
        data['Sell_Signal'] = data['Lower_Break'].astype(int) * -1  # Sell signal = -1
        data['Trade_Signal'] = data['Buy_Signal'] + data['Sell_Signal']
        return data['ATR'],data['Trade_Signal']