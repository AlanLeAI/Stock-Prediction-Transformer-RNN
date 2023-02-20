import pandas_datareader as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_stock(ticker, startday, endday):
    # Specify the start and end dates for the data
    # Download the data using the Yahoo Finance API
    data = yf.download(ticker, start=startday, end=endday)

    # Save the data to a CSV file
    data.to_csv(f"data/{ticker}_data.csv")
    return data

# Plot the close price
def plot_close_price(df):
    plt.plot(df.Close)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close Price Over Time')
    plt.show()

def preprocessing(df, hParams):
    data = df.Close.values

    # Split the data into training and test sets
    # Calculate the number of data points in each set
    train_size = int(len(data) * (1-hParams['test_prop']-hParams['valid_prop']))
    val_size = int(len(data) * hParams['valid_prop'])
    test_size = int(len(data)*hParams['test_prop'])

    # Split the data into training, validation, and test sets
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[-test_size:]

    return train_data,val_data, test_data

def generate_data(stock_prices, lookback):
    stock_prices = np.reshape(stock_prices,(-1,1))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(stock_prices)
    X, Y = [], []
    for i in range(len(scaled_data) - lookback - 1):
        x = scaled_data[i : (i + lookback)]
        X.append(x)
        Y.append(scaled_data[i + lookback])
    X, Y = np.array(X), np.array(Y)
    x_train = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return x_train, Y, scaler

def plotPredictions(df,predictions, hParams):
    #plot the data
    # print(predictions)
    train = df['Close']
    train = df[:int(len(df)-len(predictions))]
    valid = df[int(len(df)-len(predictions)):]
    valid['Predictions'] = predictions
    #visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
