import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet
from utils import plot_forecast


def main(df):
    X = df['ds']
    tscv = TimeSeriesSplit()
    train_score_list = []
    test_score_list = []
    for train_index, test_index in tscv.split(X):
        m = Prophet(weekly_seasonality=True)
        df_train = pd.DataFrame(
                {'ds': df.loc[train_index]['ds'],
                 'y': df.loc[train_index]['y']})
        m.fit(df_train)
        periods = len(test_index)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        y_train_true = df.loc[train_index]['y']
        y_train_pred = forecast.loc[train_index]['yhat']
        y_pred = forecast.loc[test_index]['yhat']
        y_true = df.loc[test_index]['y']
        train_score = mean_squared_error(y_train_true, y_train_pred)
        train_score_list.append(train_score)
        test_score = mean_squared_error(y_true, y_pred)
        test_score_list.append(test_score)
    forecast.to_csv('forecast.csv', index=False)
    print train_score_list
    print test_score_list
    plot_forecast(df_train, forecast)
    return m, np.mean(test_score_list)


if __name__ == '__main__':
    file_loc = '~/working/github/cassandra/data/retail.csv'
    df = pd.read_csv(file_loc)
    df['y'] = np.log(df['y'])
    score = main(df)
    print score[1]
