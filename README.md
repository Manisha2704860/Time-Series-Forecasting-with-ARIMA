# Time-Series-Forecasting-with-ARIMA
# What i built ?
A time series forecasting model using the ARIMA algorithm to predict future stock closing prices based on historical daily price data. The solution decomposes the time series to understand trend and seasonality, checks for stationarity, selects model parameters, fits the ARIMA model, and forecasts prices for the next 30 days with accuracy evaluation.

# Why i built it ?
To practically learn and demonstrate time series forecasting using one of the most popular statistical models, ARIMA. Forecasting stock prices is a key financial application, helping investors anticipate market trends and aid decision-making. This task also aligns with AI & ML internship requirements to analyze and predict trends using ARIMA on time series data.

# How i built it ?
- Loaded historical stock price data from a CSV file.

- Visualized the time series to understand the price movement over time.

- Applied seasonal decomposition (additive model) with a defined period to extract trend and seasonal components.

- Checked stationarity with the Augmented Dickey-Fuller (ADF) test; applied differencing if data was not stationary.

- Used ACF and PACF plots to identify suitable AR (p) and MA (q) parameters.

- Divided data into training (all but last 30 days) and testing sets (last 30 days).

- Trained an ARIMA model with order (p,d,q) set as (1,1,1).

- Generated forecasts and plotted them against actual test prices.

- Calculated Root Mean Squared Error (RMSE) for model performance evaluation.

# Installed Libraries
- pandas for data handling and CSV file reading.

- numpy for numerical computations.

- matplotlib for creating plots and visualizing data and results.

- statsmodels for performing time series decomposition, stationarity testing, ACF/PACF plotting, and ARIMA modeling.

- scikit-learn for calculating evaluation metrics (RMSE).

# Installation command:


     pip install pandas numpy matplotlib statsmodels scikit-learn
