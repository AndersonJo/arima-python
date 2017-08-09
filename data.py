import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pylab

TEST_SIZE = 50


def to_timeseries(data, lag):
    data = data.reshape(-1)
    N = len(data[lag:])
    ts = list()

    for i in range(N):
        ts.append(data[i:i + lag].tolist())
    ts = np.array(ts)
    return ts


def get_data(filename: str, lag: int):
    dataframe = pd.read_csv(filename,
                            names=['passenger'], index_col=0,
                            skiprows=1)
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe = dataframe.astype('float64')

    data = dataframe.as_matrix()
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(dataframe.values.reshape((-1, 1)))

    # Mean Adjusted Time Series
    # data_adjusted = data - np.mean(data)

    train, test = data[:-TEST_SIZE], data[-TEST_SIZE:]
    train, test = train.reshape(-1), test.reshape(-1)

    ts_data = to_timeseries(data, lag=lag)
    ts_train, ts_test = ts_data[:-TEST_SIZE], ts_data[-TEST_SIZE:]

    return dataframe, train, test, ts_train, ts_test


def diff(x: np.array, lag: int):
    return x[:-lag] - x[lag:]


def show_diff(train, test):
    x_train = np.arange(len(train))
    x_test = np.arange(len(train), len(train) + len(test))

    pylab.plot(x_train, train, color='#aaaaaa', label='train 1 lagged diff')
    pylab.plot(x_test, test, color='#666666', label='test 1 lagged diff')


if __name__ == '__main__':
    filename = 'dataset/international-airline-passengers.csv'
    dataframe, train, test, ts_train, ts_test = get_data(filename, lag=4)

    print(train.shape, ts_train.shape)
