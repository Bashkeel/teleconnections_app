import pandas as pd
import numpy as np
import json
from fbprophet.serialize import model_to_json, model_from_json
from calendar import month_abbr
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric
from fbprophet.utilities import regressor_coefficients
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def get_plotly_plots(m, forecast):

    regressors = regressor_coefficients(m)
    regressors.sort_values(by='coef', inplace=True)
    regressors.reset_index(drop=True, inplace=True)

    fig1 = go.Figure(data=go.Scatter(x=regressors.index, y=regressors['coef'], mode='markers',text=regressors['regressor']))
    fig1.update_layout(margin=dict(t=40),height=400, yaxis_title_text='Coefficient')


    fig2 = plot_plotly(m, forecast, changepoints=True, trend=True)
    start_date = "2000-01-01"
    end_date = "2015-08-01"
    fig2.update_xaxes(type="date", range=[start_date, end_date])
    fig2.update_layout(
    xaxis_title_text='Date', yaxis_title_text='Average Temperature')


    fig3 = plot_components_plotly(m, forecast)
    # fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    return fig1, fig2, fig3


def plot_monthly_averages(df):
    monthly_df = df[['ds', 'AverageTemperature']].copy(deep=True)
    monthly_df['month'] = pd.DatetimeIndex(monthly_df['ds']).month
    temp_avg = monthly_df.groupby(by=monthly_df['month']).mean()
    temp_sd = monthly_df.groupby(by=monthly_df['month']).std()

    monthly_df = pd.concat([temp_avg, temp_sd], axis=1)
    monthly_df.columns = ['Temperature', 'SD']
    monthly_df.index = list(month_abbr)[1:]
    monthly_df['Month'] = monthly_df.index
    monthly_df

    fig = px.line(monthly_df, x="Month", y="Temperature", error_y="SD")
    return fig


def plot_time_series(feature_subset, feature_selected):
    fig = px.line(feature_subset, x="Date", y="Feature")
    fig.update_layout(
        title_text=f'Historical Records of Feature: {feature_selected}')
    return fig


def plot_feature_histogram(feature_subset, feature_selected):
    fig = px.histogram(feature_subset, x="Feature", opacity=0.75)
    fig.update_layout(
        title_text=f'Distribution of Feature: {feature_selected}',
        xaxis_title_text='Value',
        yaxis_title_text='Count',
        bargap=0.2)
    return fig


def plot_feature_acf(feature_subset, feature_selected):
    fig = plot_acf(feature_subset['Feature'], lags = 50)
    plt.title(f'Autocorrelation Function for Feature: {feature_selected}')
    plt.show()
    return fig


def plot_feature_pacf(feature_subset, feature_selected):
    fig = plot_pacf(feature_subset['Feature'], lags = 50)
    plt.title(f'Partial Autocorrelation Function for Feature: {feature_selected}')
    plt.show()
    return fig


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


def plot_TLCC(feature_subset, feature_selected):
    lags = np.arange(-(0), (24), 1)
    lagged_corr_result = np.nan_to_num([crosscorr(feature_subset['Average Temperature'], feature_subset['Feature'], lag) for lag in lags])
    lagged_corr_result = pd.DataFrame(lagged_corr_result, columns=['corr'])
    lagged_corr_result['Lags'] = lags
    lagged_corr_result['abs'] = abs(lagged_corr_result['corr'])

    fig = px.line(lagged_corr_result, x="Lags", y="corr")
    # fig.add_vline(x=0)
    # fig.add_vline(x=lagged_corr_result.iloc[np.argmax(lagged_corr_result['abs']),:]['Lags'], line_dash="dash", line_color="red",
    #              annotation_text = "Peak Synchrony", annotation_font_color="red", annotation_font_size=20, annotation_position='bottom right')
    fig.update_layout(
        title_text=f'Time-Lagged Cross Correlation of Feature: {feature_selected}',
        xaxis_title_text='Lags',
        yaxis_title_text='Correlation',
    )

    fig.update_xaxes(range=[-1, 25])

    return fig
