from sklearn.base import BaseEstimator, RegressorMixin
from fbprophet import Prophet
import pandas as pd


class tsRegression(BaseEstimator, RegressorMixin):
    """ A template estimator to be used as a reference implementation .

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, growth='linear', changepoints=None, n_changepoints=25):
        self.growth = growth,
        self.changepoints = changepoints,
        self.n_changepoints = n_changepoints,
        self.yearly_seasonality = 'auto',
        self.weekly_seasonality = 'auto',
        self.holidays = None,
        self.seasonality_prior_scale = 10.0,
        self.holidays_prior_scale = 10.0,
        self.changepoint_prior_scale = 0.05,
        self.mcmc_samples = 0,
        self.interval_width = 0.95,
        self.uncertainty_samples = 1000,

    def fit(self, df):
        """A reference implementation of a fitting function

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X = df['ds'].values
        y = df['y'].values
        print(self.changepoints)

        model = Prophet(changepoints=self.changepoints,
                        n_changepoints=self.n_changepoints)
        self.df_fit = pd.DataFrame({'ds': X, 'y': y})
        model.fit(self.df_fit)
        self.model = model
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The values predicted by fbProphet.
        """
        periods = 31
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast = self.model.predict(future)
        return self.forecast['yhat']
