import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

    scaler = MinMaxScaler()
    data = scaler.fit_transform(dataframe.values.reshape((-1, 1)))

    # Mean Adjusted Time Series
    mu = np.mean(data)
    data_adjusted = data - mu

    train, test = data_adjusted[:-TEST_SIZE], data_adjusted[-TEST_SIZE:]

    ts_data = to_timeseries(data_adjusted, lag=lag)
    ts_train, ts_test = ts_data[:-TEST_SIZE], ts_data[-TEST_SIZE:]

    return dataframe, train, test, ts_train, ts_test
