import pandas as pd
import numpy as np
from calendar import month_abbr
from functools import reduce
from IPython.display import clear_output
import dateutil

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.utilities import regressor_coefficients

import os
import json
from fbprophet.serialize import model_to_json, model_from_json

from scipy.stats import normaltest
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.api import OLS


def get_city_temperature(city):
    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    return city_temp

def clean_indices(index, name):
    index.columns = list(month_abbr)
    index.rename(columns={'': 'year'}, inplace=True)
    index[index < -5] = np.nan
    reshaped_index = index.melt(id_vars=['year'])
    reshaped_index['ds'] = reshaped_index['year'].astype(str) + "-" + reshaped_index['variable'].astype(str)
    reshaped_index['ds'] = pd.to_datetime(reshaped_index['ds'], format="%Y-%b")

    ts_index = reshaped_index[['ds', 'value']].copy().sort_values('ds')
    ts_index['value'].interpolate(inplace=True)
    ts_index.rename(columns={'value': name}, inplace=True)
    return ts_index

def forecast_regressors(ts_df, months_to_predict, months_prior):
    ts_temp = ts_df.copy(deep=True)
    ts_temp.rename(columns={'AverageTemperature': 'y'}, inplace=True)
    ts_temp = ts_temp[-ts_temp['y'].isnull()]
    num_regressors = ts_temp.shape[1]-2

    m = Prophet(yearly_seasonality=True)
    if num_regressors != 0:
        for i in range(num_regressors):
            regressor = ts_temp.columns[i+2]
            m.add_regressor(regressor, prior_scale=0.5, mode='additive')
    m.fit(ts_temp)

    future = m.make_future_dataframe(periods=months_to_predict, freq="MS")
    if num_regressors != 0:
        add_regressors = ts_df.iloc[0:len(future),1:num_regressors+2]
        future = pd.concat([future, add_regressors.reset_index(drop=True)], axis=1)
    forecast = m.predict(future)

    end_date = ts_temp['ds'].max() - dateutil.relativedelta.relativedelta(months=1)*3
    start_date = ts_temp['ds'].max() - dateutil.relativedelta.relativedelta(months=1)*(months_prior+3)
    cutoffs = pd.date_range(start=start_date, end=end_date, freq='MS')

    df_cv = cross_validation(model=m, horizon='365 days', cutoffs=cutoffs)
    df_p = performance_metrics(df_cv)
    return m, forecast, df_cv


def get_temp_and_indices(city):

    # ENSO/SST Indices

    BEST = pd.read_csv('data/Indices/BEST.csv')
    BEST = clean_indices(BEST, "BEST")

    NINA = pd.read_csv('data/Indices/NINA3.csv')
    NINA = clean_indices(NINA, "NINA")

    ONI = pd.read_csv('data/Indices/ONI.csv')
    ONI = clean_indices(ONI, "ONI")


    # Teleconnections

    EA = pd.read_csv('data/Indices/EA.csv')
    EA = clean_indices(EA, "EA")

    EPNP = pd.read_csv('data/Indices/EPNP.csv')
    EPNP = clean_indices(EPNP, "EPNP")

    NAO = pd.read_csv('data/Indices/NAO.csv')
    NAO = clean_indices(NAO, "NAO")

    PNA = pd.read_csv('data/Indices/PNA.csv')
    PNA = clean_indices(PNA, "PNA")

    WP = pd.read_csv('data/Indices/WP.csv')
    WP = clean_indices(WP, "WP")

    indices = [BEST, NINA, ONI, EA, EPNP, NAO, PNA, WP]
    indices_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ds'], how='outer'), indices)

    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    ts_city = pd.merge(city_temp, indices_merged, on='ds', how='outer')
    ts_city.sort_values(by='ds', inplace=True)

    # Subset to include data from 1970 onwards
    ts_city = ts_city[ts_city['ds'] >= '1970-01-01']

    return ts_city

def get_temp_and_indices(city):

    # ENSO/SST Indices

    BEST = pd.read_csv('data/Indices/BEST.csv')
    BEST = clean_indices(BEST, "BEST")

    NINA = pd.read_csv('data/Indices/NINA3.csv')
    NINA = clean_indices(NINA, "NINA")

    ONI = pd.read_csv('data/Indices/ONI.csv')
    ONI = clean_indices(ONI, "ONI")


    # Teleconnections

    EA = pd.read_csv('data/Indices/EA.csv')
    EA = clean_indices(EA, "EA")

    EPNP = pd.read_csv('data/Indices/EPNP.csv')
    EPNP = clean_indices(EPNP, "EPNP")

    NAO = pd.read_csv('data/Indices/NAO.csv')
    NAO = clean_indices(NAO, "NAO")

    PNA = pd.read_csv('data/Indices/PNA.csv')
    PNA = clean_indices(PNA, "PNA")

    WP = pd.read_csv('data/Indices/WP.csv')
    WP = clean_indices(WP, "WP")

    indices = [BEST, NINA, ONI, EA, EPNP, NAO, PNA, WP]
    indices_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ds'], how='outer'), indices)

    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    ts_city = pd.merge(city_temp, indices_merged, on='ds', how='outer')
    ts_city.sort_values(by='ds', inplace=True)

    # Subset to include data from 1970 onwards
    ts_city = ts_city[ts_city['ds'] >= '1970-01-01']

    return ts_city

def add_diff_values(index):
    index.iloc[:,1].interpolate(inplace=True)
    index['first_difference'] = index.iloc[:,1]-index.iloc[:,1].shift(1)
    index['seasonal_difference'] = index.iloc[:,1]-index.iloc[:,1].shift(12)
    index['seasonal_first_difference'] = index['seasonal_difference']-index['seasonal_difference'].shift(1)
    colnames_to_keep = index.columns[0:1]
    colnames = [f"{index.columns[1]}_" + str(colname) for colname in index.columns]
    colnames[0:2] = index.columns[0:2]
    index.columns = colnames
    index.drop(index.columns[1], axis=1, inplace=True)
    return index


def get_temp_diff_tele_indices(city):

    # ENSO/SST Indices

    BEST = pd.read_csv('data/Indices/BEST.csv')
    BEST = clean_indices(BEST, "BEST")

    NINA = pd.read_csv('data/Indices/NINA3.csv')
    NINA = clean_indices(NINA, "NINA")

    ONI = pd.read_csv('data/Indices/ONI.csv')
    ONI = clean_indices(ONI, "ONI")

    # Teleconnections

    EA = pd.read_csv('data/Indices/EA.csv')
    EA = clean_indices(EA, "EA")
    add_diff_values(EA)

    EPNP = pd.read_csv('data/Indices/EPNP.csv')
    EPNP = clean_indices(EPNP, "EPNP")
    add_diff_values(EPNP)

    NAO = pd.read_csv('data/Indices/NAO.csv')
    NAO = clean_indices(NAO, "NAO")
    add_diff_values(NAO)

    PNA = pd.read_csv('data/Indices/PNA.csv')
    PNA = clean_indices(PNA, "PNA")
    add_diff_values(PNA)

    WP = pd.read_csv('data/Indices/WP.csv')
    WP = clean_indices(WP, "WP")
    add_diff_values(WP)

    indices = [BEST, NINA, ONI, EA, EPNP, NAO, PNA, WP]
    indices_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ds'], how='outer'), indices)

    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    ts_city = pd.merge(city_temp, indices_merged, on='ds', how='outer')
    ts_city.sort_values(by='ds', inplace=True)

    # Subset to include data from 1970 onwards
    ts_city = ts_city[ts_city['ds'] >= '1970-01-01']

    return ts_city

def add_all_lagged_values(index, name):
    index.iloc[:,1].interpolate(inplace=True)
    months_lags = [x + 1 for x in list(range(12))]
    for i in months_lags:
        col_name = name + "_t-" + str(i)
        index[col_name] = index.iloc[:,1].shift(i)
    return index

def get_temp_lagged_tele_indices(city):

    # ENSO/SST Indices

    BEST = pd.read_csv('data/Indices/BEST.csv')
    BEST = clean_indices(BEST, "BEST")

    NINA = pd.read_csv('data/Indices/NINA3.csv')
    NINA = clean_indices(NINA, "NINA")

    ONI = pd.read_csv('data/Indices/ONI.csv')
    ONI = clean_indices(ONI, "ONI")

    # Teleconnections

    EA = pd.read_csv('data/Indices/EA.csv')
    EA = clean_indices(EA, "EA")
    add_all_lagged_values(EA, 'EA')

    EPNP = pd.read_csv('data/Indices/EPNP.csv')
    EPNP = clean_indices(EPNP, "EPNP")
    add_all_lagged_values(EPNP, "EPNP")

    NAO = pd.read_csv('data/Indices/NAO.csv')
    NAO = clean_indices(NAO, "NAO")
    add_all_lagged_values(NAO, "EPNP")

    PNA = pd.read_csv('data/Indices/PNA.csv')
    PNA = clean_indices(PNA, "PNA")
    add_all_lagged_values(PNA, "PNA")

    WP = pd.read_csv('data/Indices/WP.csv')
    WP = clean_indices(WP, "WP")
    add_all_lagged_values(WP, 'WP')

    indices = [BEST, NINA, ONI, EA, EPNP, NAO, PNA, WP]
    indices_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ds'], how='outer'), indices)

    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    ts_city = pd.merge(city_temp, indices_merged, on='ds', how='outer')
    ts_city.sort_values(by='ds', inplace=True)

    # Subset to include data from 1970 onwards
    ts_city = ts_city[ts_city['ds'] >= '1970-01-01']

    return ts_city

def get_elnino_cat(index, name):
    ENSO = index.copy(deep=True).sort_values('ds')
    ENSO['Sign'] = np.sign(ENSO.iloc[:,1])
    ENSO.iloc[:,1] = ENSO.iloc[:,1].abs()
    ENSO['Rolling_Strength'] = ENSO.iloc[:,1].rolling(window=5, min_periods=5).min(key='abs')

    strength_values = [0,1,2,3,4]
    strength_conditions = [
        (ENSO['Rolling_Strength'] <= 0.5),
        (ENSO['Rolling_Strength'] > 0.5) & (ENSO['Rolling_Strength'] <= 1),
        (ENSO['Rolling_Strength'] > 1) & (ENSO['Rolling_Strength'] <= 1.5),
        (ENSO['Rolling_Strength'] > 1.5) & (ENSO['Rolling_Strength'] <= 2.0),
        (ENSO['Rolling_Strength'] > 2)
    ]

    col_name = name + "_Category"
    ENSO[col_name] = np.select(strength_conditions, strength_values)
    ENSO[col_name] = ENSO[col_name]*ENSO['Sign']
    ENSO[col_name].replace('-0', '0', inplace=True)
    ENSO.drop(ENSO.columns[2:4], axis=1, inplace=True)
    ENSO = ENSO.iloc[:,[0,2,1]]
    return ENSO


def get_temp_elnino_cat_indices(city):

    # ENSO/SST Indices
    BEST = pd.read_csv('data/Indices/BEST.csv')
    BEST = clean_indices(BEST, "BEST")
    BEST = get_elnino_cat(BEST, "BEST")

    NINA = pd.read_csv('data/Indices/NINA3.csv')
    NINA = clean_indices(NINA, "NINA")
    NINA = get_elnino_cat(NINA, "NINA")

    ONI = pd.read_csv('data/Indices/ONI.csv')
    ONI = clean_indices(ONI, "ONI")
    ONI = get_elnino_cat(ONI, "ONI")

    # Teleconnections
    EA = pd.read_csv('data/Indices/EA.csv')
    EA = clean_indices(EA, "EA")

    EPNP = pd.read_csv('data/Indices/EPNP.csv')
    EPNP = clean_indices(EPNP, "EPNP")

    NAO = pd.read_csv('data/Indices/NAO.csv')
    NAO = clean_indices(NAO, "NAO")

    PNA = pd.read_csv('data/Indices/PNA.csv')
    PNA = clean_indices(PNA, "PNA")

    WP = pd.read_csv('data/Indices/WP.csv')
    WP = clean_indices(WP, "WP")

    indices = [BEST, NINA, ONI, EA, EPNP, NAO, PNA, WP]
    indices_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ds'], how='outer'), indices)

    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    ts_city = pd.merge(city_temp, indices_merged, on='ds', how='outer')
    ts_city.sort_values(by='ds', inplace=True)

    ts_city = ts_city[ts_city['ds'] >= '1970-01-01']

    return ts_city


def get_temp_lagged_elnino(city):

    # ENSO/SST Indices

    BEST = pd.read_csv('data/Indices/BEST.csv')
    BEST = clean_indices(BEST, "BEST")
    BEST = get_elnino_cat(BEST, "BEST")
    BEST = add_all_lagged_values(BEST, "BEST_Category")

    NINA = pd.read_csv('data/Indices/NINA3.csv')
    NINA = clean_indices(NINA, "NINA")
    NINA = get_elnino_cat(NINA, "NINA")
    NINA = add_all_lagged_values(NINA, "NINA_Category")

    ONI = pd.read_csv('data/Indices/ONI.csv')
    ONI = clean_indices(ONI, "ONI")
    ONI = get_elnino_cat(ONI, "ONI")
    ONI = add_all_lagged_values(ONI, "ONI_Category")

    # Teleconnections

    EA = pd.read_csv('data/Indices/EA.csv')
    EA = clean_indices(EA, "EA")

    EPNP = pd.read_csv('data/Indices/EPNP.csv')
    EPNP = clean_indices(EPNP, "EPNP")

    NAO = pd.read_csv('data/Indices/NAO.csv')
    NAO = clean_indices(NAO, "NAO")

    PNA = pd.read_csv('data/Indices/PNA.csv')
    PNA = clean_indices(PNA, "PNA")

    WP = pd.read_csv('data/Indices/WP.csv')
    WP = clean_indices(WP, "WP")

    indices = [BEST, NINA, ONI, EA, EPNP, NAO, PNA, WP]
    indices_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ds'], how='outer'), indices)

    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    ts_city = pd.merge(city_temp, indices_merged, on='ds', how='outer')
    ts_city.sort_values(by='ds', inplace=True)

    # Subset to include data from 1970 onwards
    ts_city = ts_city[ts_city['ds'] >= '1970-01-01']

    return ts_city

def get_temp_overfitted(city):

    # ENSO/SST Indices

    BEST = pd.read_csv('data/Indices/BEST.csv')
    BEST = clean_indices(BEST, "BEST")
    BEST = get_elnino_cat(BEST, "BEST")
    BEST = add_all_lagged_values(BEST, "BEST_Category")

    NINA = pd.read_csv('data/Indices/NINA3.csv')
    NINA = clean_indices(NINA, "NINA")
    NINA = get_elnino_cat(NINA, "NINA")
    NINA = add_all_lagged_values(NINA, "NINA_Category")

    ONI = pd.read_csv('data/Indices/ONI.csv')
    ONI = clean_indices(ONI, "ONI")
    ONI = get_elnino_cat(ONI, "ONI")
    ONI = add_all_lagged_values(ONI, "ONI_Category")

    # Teleconnections

    EA = pd.read_csv('data/Indices/EA.csv')
    EA = clean_indices(EA, "EA")
    add_diff_values(EA)
    add_all_lagged_values(EA, 'EA')


    EPNP = pd.read_csv('data/Indices/EPNP.csv')
    EPNP = clean_indices(EPNP, "EPNP")
    add_diff_values(EPNP)
    add_all_lagged_values(EPNP, "EPNP")

    NAO = pd.read_csv('data/Indices/NAO.csv')
    NAO = clean_indices(NAO, "NAO")
    add_diff_values(NAO)
    add_all_lagged_values(NAO, "NAO")


    PNA = pd.read_csv('data/Indices/PNA.csv')
    PNA = clean_indices(PNA, "PNA")
    add_diff_values(PNA)
    add_all_lagged_values(PNA, "PNA")

    WP = pd.read_csv('data/Indices/WP.csv')
    WP = clean_indices(WP, "WP")
    add_diff_values(WP)
    add_all_lagged_values(WP, 'WP')

    indices = [BEST, NINA, ONI, EA, EPNP, NAO, PNA, WP]
    indices_merged = reduce(lambda  left,right: pd.merge(left,right,on=['ds'], how='outer'), indices)

    temp_major_city = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")
    city = temp_major_city[temp_major_city['City'] == city].copy()
    city['ds'] = pd.to_datetime(city['dt'], format="%Y-%m-%d")
    city['year'] = pd.DatetimeIndex(city['ds']).year
    city['month'] = pd.DatetimeIndex(city['ds']).month
    city = city[city['year'] >= 1950]
    city.reset_index(inplace=True, drop=True)
    city_temp = city[['ds', 'AverageTemperature', 'AverageTemperatureUncertainty']]
    city_temp = city_temp[city_temp['AverageTemperatureUncertainty'] < 1]
    city_temp.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
    city_temp.sort_values(by='ds')

    ts_city = pd.merge(city_temp, indices_merged, on='ds', how='outer')
    ts_city.sort_values(by='ds', inplace=True)

    # Subset to include data from 1970 onwards
    ts_city = ts_city[ts_city['ds'] >= '1970-01-01']

    return ts_city


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

def get_temp_ideal_lag(city):

    ts_city = get_temp_and_indices('Singapore')
    ts_city.drop(['ds'], axis=1, inplace=True)
    ts_city = ts_city[-ts_city['AverageTemperature'].isnull()]
    ideal_lags = []
    lags = np.arange(-(1), (24), 1)
    lagged_corr_df = pd.DataFrame(lags, columns=['Lags'])
    for i in range(ts_city.shape[1]-1):
        lagged_corr_result = np.nan_to_num([crosscorr(ts_city.iloc[1:,0], ts_city.iloc[1:,i+1], lag) for lag in lags])
        lagged_corr_result = pd.DataFrame(lagged_corr_result, columns=[ts_city.columns[i+1]])
        lagged_corr_result['Lags'] = lags
        lagged_corr_result['abs'] = abs(lagged_corr_result.iloc[:,0])
        lagged_corr_df = pd.concat([lagged_corr_df, lagged_corr_result], axis=1)
        ideal_lags.append(lagged_corr_result["Lags"][np.argmax(lagged_corr_result["abs"])])

    ts_city = get_temp_and_indices(city)
    ts_city['BEST'] = ts_city['BEST'].shift(ideal_lags[0])
    ts_city['NINA'] = ts_city['NINA'].shift(ideal_lags[1])
    ts_city['ONI'] = ts_city['ONI'].shift(ideal_lags[2])
    ts_city['EA'] = ts_city['EA'].shift(ideal_lags[3])
    ts_city['EPNP'] = ts_city['EPNP'].shift(ideal_lags[4])
    ts_city['NAO'] = ts_city['NAO'].shift(ideal_lags[5])
    ts_city['PNA'] = ts_city['PNA'].shift(ideal_lags[6])
    ts_city['WP'] = ts_city['WP'].shift(ideal_lags[7])
    ts_city = ts_city.iloc[max(ideal_lags):,]

    return ts_city


def get_summary_statistics(df):

    df_nodate = df.iloc[:,1:].copy(deep=True)

    summary_stats = pd.DataFrame([])

    # Overall Pearson Correlation
    summary_stats['Correlation'] = df.corr().iloc[0,:]

    # Normality Test
    normal_pvals = []
    for i in range(df.shape[1]-1):
        zscore,p = normaltest(df.iloc[:,i+1])
        normal_pvals.append(p)
    summary_stats['Normality'] = normal_pvals

    # Stationarity
    adf_pvals = []
    for i in range(df.shape[1]-1):
        results = adfuller(df.iloc[:,i+1], autolag='AIC')
        adf_pvals.append(results[1])
    summary_stats['Stationary_ADF'] = adf_pvals

    kpss_pvals = []
    for i in range(df.shape[1]-1):
        statistic, p_value, n_lags, critical_values = kpss(df.iloc[:,i+1], nlags='auto')
        kpss_pvals.append(p_value)
    summary_stats['Stationarity_KPSS'] = kpss_pvals

    # Co-Integration
    EG_pvals = []
    for i in range(df_nodate.shape[1]):
        ols_result = OLS(df_nodate['AverageTemperature'], df_nodate.iloc[:,i]).fit()
        result = adfuller(ols_result.resid)
        EG_pvals.append(results[1])
    summary_stats['Co-integration'] = EG_pvals

    maxlag=24
    def grangers_causality_matrix(data, variables, test = 'ssr_chi2test', verbose=False):
        p_dataset = pd.DataFrame(np.zeros((1, len(variables))), columns=variables, index=['x'])
        lag_dataset = pd.DataFrame(np.zeros((1, len(variables))), columns=variables, index=['x'])

        for c in df_nodate.columns[1:]:
            test_result = grangercausalitytests(data[['AverageTemperature', c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_lag = np.argmax([round(test_result[i+1][0]['ssr_chi2test'][1],4) for i in range(maxlag)])
            min_p_value = np.max(p_values)

            p_dataset[c] = min_p_value
            lag_dataset[c] = min_lag+1

        return p_dataset, lag_dataset

    granger_p_mat, granger_lag_mat = grangers_causality_matrix(df_nodate.iloc[1:,], variables = df_nodate.columns)

    summary_stats['Granger_Causality_pvals'] = list(granger_p_mat[0:1].transpose().iloc[:,0])
    summary_stats['Granger_Causality_Lags'] = list(granger_lag_mat[0:1].transpose().iloc[:,0])

    # Synchrony
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


    ideal_lags = []
    lags = np.arange((0), (24), 1)
    for i in range(df.shape[1]-1):
        lagged_corr_result = np.nan_to_num([crosscorr(df.iloc[:,1], df.iloc[:,i+1], lag) for lag in lags])
        lagged_corr_result = pd.DataFrame(lagged_corr_result, columns=[df.columns[i+1]])
        lagged_corr_result['Lags'] = lags
        lagged_corr_result['abs'] = abs(lagged_corr_result.iloc[:,0])
        ideal_lags.append(lagged_corr_result["Lags"][np.argmax(lagged_corr_result["abs"])])

    summary_stats['TLCC_Lags'] = ideal_lags

    summary_stats = summary_stats.transpose()
    return summary_stats
