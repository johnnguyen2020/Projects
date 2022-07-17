#!/usr/bin/env python
# coding: utf-8

# 
# I affirm that I will not plagiarize, use unauthorized materials, or give or receive illegitimate help on assignments, papers, or examinations. I will also uphold equity and honesty in the evaluation of my work and the work of others. I do so to sustain a community built around this Code of Honor.
# 
# By filling out the following fields, you are signing this pledge.  No assignment will get credit without being pledged.
# 
# Name: John Nguyen
# 
# ID: jn2814
# 
# Date: 06/31/2022
# 
# 
# # Final Project

# In[680]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from tqdm import tqdm
import pypfopt
from pypfopt import risk_models, expected_returns, plotting
pypfopt.__version__ 

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogarithmicError
get_ipython().system('pip install -q -U keras-tuner')
import keras_tuner as kt

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# # Investment Strategy: Picking 10 Stocks 

# In[681]:


tickers = ["GE","CAT","AMGN","QCOM","CSCO","FB","TXN","LRCX","AMAT","IBM","ORCL","JNJ","HON","MRK","GOOGL",            "GOOG","MDT","TMO","AAPL","ABBV","ABT","MSFT","ADP","RTX","DHR","LLY","ACN","TMUS","AVGO","AMD",            "PYPL","SYK","NFLX","ZTS","ISRG","INTU","NVDA","CRM","T","NOW",]


# In[682]:


import yfinance as yf
endd = '2022-07-01'
startd = '2022-06-01'
ohlc = yf.download(tickers, start=startd, end=endd)
#prices = ohlc.pop("Adj Close")
#prices = ohlc["Adj Close"].dropna(how="all")
#prices.plot(figsize=(15,10));


# In[683]:


prices = ohlc["Adj Close"].dropna(how="all")
prices


# In[684]:


prices.plot(figsize=(15,10));


# In[685]:


plt.figure(figsize=(15,15));
sns.heatmap(data=prices.corr(),vmin=-1,vmax=1,linewidths=.3,annot=True, cmap=plt.cm.coolwarm,square=True);
correlation_matrix = prices.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool))
drop_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] >= 0.9)]
print("Whole Period Highly Correlated Features: " ,drop_columns)


# In[686]:


plt.figure(figsize=(25,25));
sns.heatmap(data=abs(prices.corr())>.9,vmin=0,vmax=1,linewidths=.3, cmap=plt.cm.Reds,square=True);


# In[687]:


#non_correlated = tickers - drop_columns
def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

print(Diff(tickers, drop_columns))


# In[688]:


#stocks of interest + stock from each sector with highest market cap + stocks from each sector with highest PE Ratio
tickers_16 = ['ORCL', 'AAPL', 'ABBV', 'IBM', 'RTX', 'AMGN'] +              ['GOOGL', 'SYK', 'LRCX', 'GE', 'TMUS'] +              ['NOW', 'T', 'NVDA', 'ISRG', 'HON']


# In[689]:


endd = '2022-06-13'
startd = '2022-05-01'
ohlc_16 = yf.download(tickers_16, start=startd, end=endd)
prices_16 = ohlc_16["Adj Close"].dropna(how="all")
prices_16


# In[690]:


prices_16.plot(figsize=(15,10));


# In[691]:


symbol = 'ORCL'
data = pd.DataFrame(prices_16[symbol])
data.rename(columns={symbol: 'price'}, inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_scaled=scaler.fit_transform(data.to_numpy())
data['Scaled Price'] = data_scaled


# In[692]:


lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['Scaled Price'].shift(lag) # <1>
    cols.append(col)
data.dropna(inplace=True)


# In[693]:


data


# In[694]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

og_label = data.pop('price')
label = data.pop('Scaled Price')

X_train, X_test, y_train, y_test = train_test_split(data,label, train_size=0.8, shuffle = False, stratify = None)


# In[695]:


X_train


# In[696]:


_X_train = X_train_.to_numpy().reshape(X_train_.shape[0], X_train_.shape[1], 1)
_X_test_ = X_test_.to_numpy().reshape(X_test_.shape[0], X_test_.shape[1], 1)


# In[697]:


def build_model(hp):
    model_opt = Sequential()
  
    hp_units1 = hp.Int('units1', min_value=1, max_value=10, step=2)
    hp_units2 = hp.Int('units2', min_value=1, max_value=10, step=2)
    hp_units3 = hp.Int('units3', min_value=1, max_value=10, step=2)
    
    model_opt.add(LSTM(units=7,return_sequences=True, input_shape=(5,1)))
    model_opt.add(LSTM(units=7,return_sequences=True))
    model_opt.add(LSTM(units=7,return_sequences=False))
    #model_opt.add(Dropout(hp.Float('dropout', min_value=0, max_value=0.5, step=0.1)))
    model_opt.add(Dense(1))
    
    model_opt.compile(optimizer='adam',loss='mse',metrics=['mean_squared_error'])

    return model_opt

tuner = kt.RandomSearch(build_model,objective='mean_squared_error',seed=42,max_trials=10, overwrite=True)
tuner.search(_X_train, y_train, epochs=100)


# In[698]:


for h_param in [f"units{i}" for i in range(1,4)]:
  print(h_param, tuner.get_best_hyperparameters()[0].get(h_param))


# In[699]:


best_model = tuner.get_best_models(num_models=1)[0]
loss, mse = best_model.evaluate(_X_test_, y_test)


# In[700]:


best_model.summary()


# In[701]:


train_predict=best_model.predict(_X_train)
test_predict=best_model.predict(_X_test_)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
math.sqrt(mean_squared_error(y_train,train_predict))


# In[702]:


predictions = np.append(train_predict, test_predict)


# In[703]:


data['price'] = og_label
data['predictions'] = predictions
data[['price', 'predictions']].plot(figsize=(10, 6));
plt.title(symbol)


# In[704]:


data.tail()


# In[705]:


data_future = data.copy()
data_future.pop('predictions')
data_future['scaled price'] = label
data_future.tail()


# In[706]:


temp = data_future.iloc[data_future.shape[0]-6:data_future.shape[0]-1, 0:5].to_numpy().reshape(5,5,1)
temp


# In[707]:


temp_predict=best_model.predict(temp)
temp_predict = scaler.inverse_transform(temp_predict)
temp_predict


# In[708]:


temp_predict[4][0]


# In[709]:


forcast_period = 7
for i in range (0,forcast_period):
    temp = data_future.iloc[data_future.shape[0]-6:data_future.shape[0]-1, 0:5].to_numpy().reshape(5,5,1)
    temp_predict=best_model.predict(temp)
    
    row_temp = {'lag_1': data_future['scaled price'][data_future.shape[0]-2],             'lag_2': data_future['scaled price'][data_future.shape[0]-3],             'lag_3': data_future['scaled price'][data_future.shape[0]-4],             'lag_4': data_future['scaled price'][data_future.shape[0]-5],             'lag_5': data_future['scaled price'][data_future.shape[0]-6],             'scaled price': temp_predict[4][0], 'price': scaler.inverse_transform(temp_predict)[4][0]}
    
    data_future = data_future.append(row_temp, ignore_index = True)
    
#data_future.drop([data_future.shape[0]-forcast_period], axis=0, inplace=True)


# In[710]:


data_future


# In[711]:


seven_day_return = data_future.iloc[data_future.shape[0]-1, 5] - data_future.iloc[data_future.shape[0]-8, 5]


# In[712]:


print(symbol, seven_day_return)


# In[714]:


seven_day_return = []

for ticker in tickers_16:
    data = pd.DataFrame(prices_16[ticker])
    data.rename(columns={ticker: 'price'}, inplace=True)

    scaler=MinMaxScaler()
    data_scaled=scaler.fit_transform(data.to_numpy())
    data['Scaled Price'] = data_scaled

    lags = 5
    cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data[col] = data['Scaled Price'].shift(lag) # <1>
        cols.append(col)
    data.dropna(inplace=True)

    og_label = data.pop('price')
    label = data.pop('Scaled Price')

    X_train, X_test, y_train, y_test = train_test_split(data,label, train_size=0.8, shuffle = False, stratify = None)

    _X_train = X_train_.to_numpy().reshape(X_train_.shape[0], X_train_.shape[1], 1)
    _X_test_ = X_test_.to_numpy().reshape(X_test_.shape[0], X_test_.shape[1], 1)
    
    tuner = kt.RandomSearch(build_model,objective='mean_squared_error',seed=42,max_trials=10, overwrite=True)
    tuner.search(_X_train, y_train, epochs=10)
    
    best_model = tuner.get_best_models(num_models=1)[0]

    train_predict=best_model.predict(_X_train)
    test_predict=best_model.predict(_X_test_)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    
    predictions = np.append(train_predict, test_predict)
    
    data['price'] = og_label
    data['predictions'] = predictions
    
    '''
    data[['price', 'predictions']].plot(figsize=(10, 6));
    plt.title(ticker)
    '''
    
    data_future = data.copy()
    data_future.pop('predictions')
    data_future['scaled price'] = label
    
    forcast_period = 7
    for i in range (0,forcast_period):
        temp = data_future.iloc[data_future.shape[0]-6:data_future.shape[0]-1, 0:5].to_numpy().reshape(5,5,1)
        temp_predict=best_model.predict(temp)

        row_temp = {'lag_1': data_future['scaled price'][data_future.shape[0]-2],                 'lag_2': data_future['scaled price'][data_future.shape[0]-3],                 'lag_3': data_future['scaled price'][data_future.shape[0]-4],                 'lag_4': data_future['scaled price'][data_future.shape[0]-5],                 'lag_5': data_future['scaled price'][data_future.shape[0]-6],                 'scaled price': temp_predict[4][0], 'price': scaler.inverse_transform(temp_predict)[4][0]}

        data_future = data_future.append(row_temp, ignore_index = True)
        
    seven_day_return_temp = data_future.iloc[data_future.shape[0]-1, 5] - data_future.iloc[data_future.shape[0]-8, 5]
    seven_day_return.append(seven_day_return_temp)


# In[716]:


df_results = pd.DataFrame({'Stocks':tickers_16})
df_results['7 day Return'] = seven_day_return


# In[829]:


df_results.sort_values('7 day Return', ascending=False).head(10)


# In[720]:


df_results.sort_values('7 day Return').head(5)


# In[830]:


gold_stocks = ['ISRG', 'SYK', 'AAPL', 'HON', 'ABBV'] + ['GOOGL', 'NOW', 'LRCX', 'AMGN', 'TMUS']
gold_stocks = ['ISRG', 'SYK', 'AAPL', 'HON', 'ABBV', 'GE', 'ORCL', 'T', 'RTX', 'NVDA'] 


# # Investment Strategy: Assest Allocation

# In[918]:


endd = '2022-07-13'
startd = '2022-06-13'
gold_stocks = ['ISRG', 'SYK', 'AAPL', 'HON', 'ABBV', 'GE', 'ORCL', 'T', 'RTX', 'NVDA'] 
ohlc = yf.download(gold_stocks, start=startd, end=endd)
prices = ohlc["Adj Close"].dropna(how="all")
prices.tail()


# In[919]:


ohlc_all = yf.download(gold_stocks, period="max")
prices_all = ohlc_all["Adj Close"].dropna(how="all")
prices_all[prices_all.index >= startd].plot(figsize=(15,10));
plt.title('Selected Stock Prices since: ' + startd)


# In[920]:


from pypfopt import risk_models
from pypfopt import plotting
from pypfopt import expected_returns

sample_cov = risk_models.sample_cov(prices, frequency=252)
sample_cov


# In[921]:


S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
plotting.plot_covariance(S, plot_correlation=True);


# In[922]:


mu = expected_returns.capm_return(prices)
mu


# In[923]:


from pypfopt import EfficientFrontier

ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

'''
AAPL_index = ef.tickers.index("AAPL")
ef.add_constraint(lambda w: w[AAPL_index] <= 0.10)
ef.add_constraint(lambda w: w[AAPL_index] >= 0.05)

ABBV_index = ef.tickers.index("ABBV")
ef.add_constraint(lambda w: w[ABBV_index] <= 0.10)
ef.add_constraint(lambda w: w[ABBV_index] >= 0.05)

AMGN_index = ef.tickers.index("AMGN")
ef.add_constraint(lambda w: w[AMGN_index] <= 0.10)
ef.add_constraint(lambda w: w[AMGN_index] >= 0.05)

GOOGL_index = ef.tickers.index("GOOGL")
ef.add_constraint(lambda w: w[GOOGL_index] <= 0.10)
ef.add_constraint(lambda w: w[GOOGL_index] >= 0.05)

HON_index = ef.tickers.index("HON")
ef.add_constraint(lambda w: w[HON_index] <= 0.10)
ef.add_constraint(lambda w: w[HON_index] >= 0.05)

ISRG_index = ef.tickers.index("ISRG")
ef.add_constraint(lambda w: w[ISRG_index] <= 0.10)
ef.add_constraint(lambda w: w[ISRG_index] >= 0.05)

LRCX_index = ef.tickers.index("LRCX")
ef.add_constraint(lambda w: w[LRCX_index] <= 0.10)
ef.add_constraint(lambda w: w[LRCX_index] >= 0.05)

NOW_index = ef.tickers.index("NOW")
ef.add_constraint(lambda w: w[NOW_index] <= 0.10)
ef.add_constraint(lambda w: w[NOW_index] >= 0.05)

SYK_index = ef.tickers.index("SYK")
ef.add_constraint(lambda w: w[SYK_index] <= 0.10)
ef.add_constraint(lambda w: w[SYK_index] >= 0.05)

TMUS_index = ef.tickers.index("TMUS")
ef.add_constraint(lambda w: w[TMUS_index] <= 0.10)
ef.add_constraint(lambda w: w[TMUS_index] >= 0.05)

'''


#ef.min_volatility()
ef.max_sharpe()

weights = ef.clean_weights()
weights


# In[924]:


pd.Series(weights).plot.barh();


# In[925]:


ef.portfolio_performance(verbose=True);


# In[927]:


from pypfopt import DiscreteAllocation

latest_prices = prices.iloc[-1]
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=1000000)
alloc, leftover = da.greedy_portfolio()
alloc


# # Performance Metrics

# In[906]:


endd = '2022-07-13'
startd = '2022-06-01'
gold_stocks = gold_stocks + ['^GSPC'] + ['^TNX']
ohlc = yf.download(gold_stocks, start=startd, end=endd)
prices = ohlc["Adj Close"].dropna(how="all")
prices


# In[ ]:




