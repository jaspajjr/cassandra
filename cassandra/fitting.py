from fbprophet import Prophet


def fit_model(df, param_dict, periods):
    m = Prophet(**param_dict)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast
