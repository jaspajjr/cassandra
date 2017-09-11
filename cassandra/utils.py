import matplotlib.pyplot as plt


def plot_forecast(training_df, fcst, ax=None, uncertainty=True, plot_cap=True,
                  xlabel='ds', ylabel='y'):
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    print(training_df.head())
    ax.plot_date(fcst['ds'], fcst['yhat'], ls='-', c='#0072B2')
    ax.plot_date(training_df['ds'], training_df['y'], c='green')
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()
