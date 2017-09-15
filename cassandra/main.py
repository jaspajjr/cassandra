import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, _search
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet
from utils import fit_model, plot_forecast


def fit_params(param_dict, df):
    X = df['ds']
    tscv = TimeSeriesSplit()
    train_score_list = []
    test_score_list = []
    for train_index, test_index in tscv.split(X):
        m = Prophet(**param_dict)
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
        train_score_list.append(mean_squared_error(y_train_true, y_train_pred))
        test_score_list.append(mean_squared_error(y_true, y_pred))
        evaluation = {
                'train_score': np.mean(train_score_list),
                'test_score': np.mean(test_score_list)}
    return evaluation


def grid_search(df, param_dict):
    optimum_score = np.inf
    for param in _search.ParameterGrid(param_dict):
        try:
            score = fit_params(param, df)
        except RuntimeError:
            score['test_score'] = np.inf
        if score['test_score'] < optimum_score:
            optimum_score = score['test_score']
            best_params = param
    if np.isinf(optimum_score):
        raise
    return {'score': optimum_score,
            'params': best_params}


def main(df):
    param_dict = {
        'weekly_seasonality': [False, True],
        'seasonality_prior_scale': [.01, .1, 1, 10, 100],
        'changepoint_prior_scale': [.01, .1, 1, 10, 100],
        'n_changepoints': [1, 5, 10, 25, 50]
        }
    result = grid_search(df, param_dict)
    model = fit_model(df, result['params'])
    plot_forecast(model)


if __name__ == '__main__':
    file_loc = '~/working/github/cassandra/data/retail.csv'
    df = pd.read_csv(file_loc)
    df['y'] = np.log(df['y'])
    main(df)
