import quandl
import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'zD6Adeqgd3n3CfJMiWj6'
#Kind of service WIKI 
#dataframe
df = quandl.get('WIKI/TSLA')
df = df[['Adj. Close']]

forecast = 30
df['Prediction'] = df[['Adj. Close']].shift(-forecast)
#Features 
X = np.array(df.drop(['Prediction'],1))
#Mean 0 
#Standar deviation 1 
X = preprocessing.scale(X)
X_forecast = X[-forecast:]
X = X[:-forecast]

y = np.array(df['Prediction'])
y = y[:-forecast]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#estimator instance (classifier) 
clf = LinearRegression()
clf.fit(X_train, y_train)
#p-value(confidence)
confidence =clf.score(X_test, y_test)

forecast_predicted = clf.predict(X_forecast)

dates = pd.date_range(start='2018-03-28', end='2018-04-26')
plt.plot(dates, forecast_predicted, color='y')
df['Adj. Close'].plot(color='g')
plt.xlim(xmin=datetime.date(2017,4,26))
plt.show()
