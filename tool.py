import numpy as np
import pylab
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def show_result(train: np.array, test: np.array, train_pred: np.array, test_pred: np.array, lag: int):
    x_train = np.arange(len(train_pred))
    x_test = np.arange(len(train_pred), len(train_pred) + len(test_pred))

    pylab.ylim(np.min(train), np.max(test) + np.std(test) * 1.3)
    pylab.plot(x_train, train[lag:], linestyle="-", label='train_true', color="#555555")
    pylab.plot(x_train, train_pred, label='train_pred', color='blue')
    pylab.plot(x_test, test, label='test_true', color='#555555')
    pylab.plot(x_test, test_pred, label='pred_test', color='red')
    pylab.legend()
    pylab.grid()

    print('Lag Value:', lag)
    print('Train R^2:', r2_score(train[lag:], train_pred))
    print('Train MSE:', mean_squared_error(train[:-lag], train_pred))
    print('Test  R^2:', r2_score(test, test_pred))
    print('Test  MSE:', mean_squared_error(test, test_pred))
