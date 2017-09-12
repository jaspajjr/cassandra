import pandas as pd
from forecast import tsRegression
from fbprophet import Prophet
from utils import cv_rmse


def main(df):
    ts = tsRegression()
    score = cv_rmse(ts, df)
    return score


if __name__ == '__main__':
    file_loc = '~/working/github/cassandra/data/retail.csv'
    df = pd.read_csv(file_loc)
    score = main(df)
