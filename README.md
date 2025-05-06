# Ex.No: 08 MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING

## AIM:

To implement Moving Average Model and Exponential smoothing Using Python.

## ALGORITHM:

Import necessary libraries

Read the CSV file.

Display the shape and the first 10 rows of the dataset

Perform rolling average transformation with a window size of 5 and 10

Display first 10 and 20 values repecively and plot them both

Perform exponential smoothing and plot the fitted graph and orginal graph

## PROGRAM:

### NAME : Bhuvaneshwaran H
### REG NO: 212223240018

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
data = pd.read_csv('silver.csv')

rate_data = data[['USD']]

print("Shape of the dataset:", rate_data.shape)
print("First 10 rows of the dataset:")
print(rate_data.head(10))

plt.figure(figsize=(12, 6))
plt.plot(rate_data['USD'], label='Original USD data')
plt.title('Original USD Data')
plt.xlabel('Rate')
plt.ylabel('USD')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = rate_data['USD'].rolling(window=5).mean()
rolling_mean_10 = rate_data['USD'].rolling(window=10).mean()
rolling_mean_5.head(10)
rolling_mean_10.head(20)

plt.figure(figsize=(12, 6))
plt.plot(rate_data['USD'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of USD')
plt.xlabel('Rate')
plt.ylabel('USD')
plt.legend()
plt.grid()
plt.show()

data.head()

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_monthly = data.resample('MS').mean()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_usd = pd.Series(
    scaler.fit_transform(data_monthly[['USD']]).flatten(),
    index=data_monthly.index,
    name='USD_scaled'
)

scaled_usd=scaled_usd+1  
x=int(len(scaled_usd)*0.8)
train_data = scaled_usd[:x]
test_data = scaled_usd[x:]
from sklearn.metrics import mean_squared_error

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_usd.var()),scaled_usd.mean()

usd_series = data_monthly['USD']

model = ExponentialSmoothing(usd_series, trend='add', seasonal='mul', seasonal_periods=12).fit()

predictions = model.forecast(steps=int(len(usd_series)/4))

ax = usd_series.plot(figsize=(10, 5))
predictions.plot(ax=ax)
ax.legend(["USD monthly", "USD forecast"])
ax.set_xlabel('Date')
ax.set_ylabel('USD Rate')
ax.set_title('PREDICTION')
plt.show()
```

## OUTPUT:

Original data:

![image](https://github.com/user-attachments/assets/9cfceaf2-9ff2-46c1-aac0-89ec3ca1ad87)

![image](https://github.com/user-attachments/assets/d6b5b482-484e-4d8d-b8fc-69ca9f7f4c4c)

Moving Average:- (Rolling)

![image](https://github.com/user-attachments/assets/952e0702-4323-438b-bea4-4ad86647a298)

Plot:

![image](https://github.com/user-attachments/assets/5e5928cd-7ac2-4589-818c-1e8ae71a0229)

Exponential Smoothing:

Test: 

![image](https://github.com/user-attachments/assets/7963a505-0ba3-4a97-b21d-2b16ccb3b24c)

Performance: (MSE)

![image](https://github.com/user-attachments/assets/2af67781-95cd-415e-ac40-85cc545bb3e2)

Prediction:

![image](https://github.com/user-attachments/assets/4a0e176d-3e7d-4485-a4ba-10553fa73d4d)

## RESULT:

Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
