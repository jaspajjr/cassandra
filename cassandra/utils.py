import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import sklearn.metrics as skm


def plot_forecast(training_df, fcst, ax=None, uncertainty=True, plot_cap=True,
                  xlabel='ds', ylabel='y'):
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ax.plot_date(fcst['ds'], fcst['yhat'], ls='-', c='#0072B2')
    ax.plot_date(training_df['ds'], training_df['y'], c='green')
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()


def build_train_df(X, y, train_index, test_index):
    X_train, _ = X[train_index], X[test_index]
    y_train, _ = y[train_index], y[test_index]  # noqa
    df_fit = pd.DataFrame({'ds': X_train, 'y': y_train})
    return df_fit


def build_outcome_df(X, df_fit, forecast, test_index):
    pred_df_ds = pd.DataFrame({'ds': X[test_index]})
    outcome_df = forecast.join(pred_df_ds, how='inner', rsuffix='_'). \
        join(df_fit, how='inner', rsuffix='__')
    return outcome_df


def ts_fit_predict(model, X, y, train_index, test_index):
    df_fit = build_train_df(X, y, train_index, test_index)
    model.fit(df_fit)
    future = model.make_future_dataframe(periods=len(test_index))
    forecast = model.predict(future)
    outcome_df = build_outcome_df(X, df_fit, forecast, test_index)
    y_true = outcome_df['y']
    y_pred = outcome_df['yhat']
    return y_true, y_pred


def cv_rmse(model, df):
    df_fit = df
    X = df_fit['ds'].values
    y = df_fit['y'].values
    rmse_list = []
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=3)
    for train_index, test_index in tscv.split(X):
        y_true, y_pred = ts_fit_predict(model, X, y, train_index, test_index)
        rmse_list.append(skm.mean_squared_error(y_true, y_pred))
    return np.mean(rmse_list)
