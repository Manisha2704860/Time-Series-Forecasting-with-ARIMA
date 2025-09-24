# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# 1. Load your stock price data
# Download historical prices (e.g., from Yahoo Finance) and save as 'stock_data.csv'
df = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')
ts = df['Close']

# 2. Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.title('Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# 3. Decompose trend/seasonality
decomp = seasonal_decompose(ts, model='additive', period=15)
decomp.plot()
plt.show()

# 4. Check stationarity with ADF test
adf_result = adfuller(ts.dropna())
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])

# 5. Differencing if not stationary
if adf_result[1] > 0.05:
    ts_diff = ts.diff().dropna()
    print('Used first differencing, plot below:')
    plt.plot(ts_diff)
    plt.title('First Difference of Closing Price')
    plt.show()
else:
    ts_diff = ts.dropna()

# 6. ACF and PACF plot for ARIMA(p,d,q) selection
plot_acf(ts_diff)
plt.show()
plot_pacf(ts_diff)
plt.show()

# 7. Train-test split (last 30 days for test)
train = ts[:-30]
test = ts[-30:]

# 8. Fit ARIMA Model (adjust order according to your ACF/PACF)
# Example uses ARIMA(1,1,1)
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# 9. Forecast and plot
forecast = model_fit.forecast(steps=30)
plt.figure(figsize=(10,4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.legend()
plt.show()

# 10. Evaluate (optional: print RMSE)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test, forecast))
print('Test RMSE:', rmse)