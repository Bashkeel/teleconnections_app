import streamlit as st
import pydeck as pdk
from src.helpers import *
from src.plots import *

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


st.set_page_config(page_icon="☀️",
    page_title="Teleconnection Temperature Forecasting",
    initial_sidebar_state="expanded",
    layout='wide')


st.sidebar.header("Temperature Forecasting Using Teleconnections & El Nino Indices")
df = pd.read_csv('data/City_Coordinates.csv')


coordinates = pd.read_csv("data/City_Coordinates.csv")
models = pd.read_csv("data/model_list.csv")
feature_list = pd.read_csv("data/feature_list_models.csv")
cities = ['Singapore', 'Tokyo', 'Toronto', 'Delhi', 'Cairo', 'Riyadh', 'Los Angeles', 'Lagos', 'Paris', 'Moscow']


city_list = st.sidebar.selectbox("Choose a city", cities, index=0, key='cities')
model_list = st.sidebar.selectbox("Select a model", models['dropdown'], key='models')
selected_model = models[models['dropdown'] == model_list].iloc[0,0]
# selected_model = models[models['dropdown'] == model_list].iloc[0,0]
if model_list:
    feature_list = pd.read_csv("data/feature_list_models.csv")
    features = feature_list[feature_list['dropdown'] == model_list].copy(deep=True)
    test = list(features['feature'])
    feature_selected = st.sidebar.selectbox('Select a feature to analyze: ', test, index=0, key='features')
run_forecast = st.sidebar.button(label='Run the Forecast')

selectedcity = df[df['City'] == city_list]

layer = pdk.Layer(
    "ScatterplotLayer",
    df,
    pickable=True,
    # auto_highlight=True,
    opacity=0.8,
    filled=True,
    radius_scale=2,
    radius_min_pixels=5,
    radius_max_pixels=50,
    line_width_min_pixels=0.01,
    get_position='[Longitude, Latitude]',
    get_fill_color=[100, 100, 200],
    get_line_color=[0, 0, 0],
)

layer_selected = pdk.Layer(
    "ScatterplotLayer",
    selectedcity,
    pickable=True,
    # auto_highlight=True,
    opacity=0.8,
    filled=True,
    radius_scale=2,
    radius_min_pixels=8,
    radius_max_pixels=50,
    line_width_min_pixels=0.01,
    get_position='[Longitude, Latitude]',
    get_fill_color=[200, 100, 100],
    get_line_color=[0, 0, 0],
)

# Set the viewport location
view_state = pdk.ViewState(latitude=selectedcity['Latitude'].iloc[-1], longitude=selectedcity['Longitude'].iloc[-1], zoom = 3, min_zoom= 0, max_zoom=5)

# Render
r = pdk.Deck(layers=[layer, layer_selected], map_style='mapbox://styles/mapbox/light-v10',
             initial_view_state=view_state,
             tooltip={"html": "<b>City Name: </b> {City} <br /> "
             "<b>Country: </b> {Country} <br /> "
             "<b>Longitude: </b> {Longitude} <br /> "
             "<b>Latitude: </b> {Latitude} <br /> ",
             })

st.sidebar.pydeck_chart(r)

if run_forecast:
    # # Quick fix: will fix later
    # if selected_model == "ideal_lag":
    #     selected_model = 'TLCC Ideal Lag'
    stats_table = pd.read_csv(f'data/models/{city_list}/{selected_model}/{selected_model}_stats.csv')
    stats_table.index = stats_table.iloc[:,0]
    stats_table.drop(columns=stats_table.columns[0], axis=1, inplace=True)

    model_table = pd.read_csv(f'data/models/{city_list}/summary_table.csv')
    model_table.index = model_table['Model']
    model_table = model_table.iloc[:,1:-1]

    df = pd.read_csv(f'data/models/{city_list}/{selected_model}/{selected_model}_data.csv')
    df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d")
    feature_subset = feature_subset = df[['ds', 'AverageTemperature', feature_selected]]
    feature_subset.columns = ['Date', 'Average Temperature', "Feature"]

    with open(f'data/models/{city_list}/{selected_model}/m.json', 'r') as fin:
        m = model_from_json(json.load(fin))  # Load model

    forecast = pd.read_csv(f"data/models/{city_list}/{selected_model}/forecast.csv")
    forecast['ds'] = pd.to_datetime(forecast['ds'], format="%Y-%m-%d")

    regressors_plot, forecast_plot, components_plot = get_plotly_plots(m, forecast)


    plot_cols = st.beta_columns(2)
    with plot_cols[0]:
        st.header('Cross Validation Results of All Models')
        st.markdown("##")
        st.markdown("##")
        st.write(model_table)

    with plot_cols[1]:
        st.header("Historical Monthly Averages")
        st.write(plot_monthly_averages(df), use_container_width=True)


    plot_cols = st.beta_columns(2)
    with plot_cols[0]:
        st.header(f"Temperature Forecast for {city_list}")
        st.plotly_chart(forecast_plot, use_container_width=True)

    with plot_cols[1]:
        st.header(f'Component Analysis of the {model_list} Model')
        st.markdown("#")
        st.plotly_chart(components_plot, use_container_width=True)


    st.header(f'Regressor Coefficients of {model_list} Model')
    st.plotly_chart(regressors_plot, use_container_width=True)

    st.header(f"Statistical Summary of Features in {model_list} Model")
    st.write(stats_table)

    st.markdown("##")
    st.markdown("##")
    st.header(f'Feature-Specific Analysis: {feature_selected}')
    st.markdown("##")
    plot_cols = st.beta_columns(2)
    with plot_cols[0]:
        st.write(plot_time_series(feature_subset, feature_selected))
        st.write(plot_feature_acf(feature_subset, feature_selected), use_container_width=True)

    with plot_cols[1]:
        st.write(plot_feature_histogram(feature_subset, feature_selected))
        st.write(plot_feature_pacf(feature_subset, feature_selected))

    st.plotly_chart(plot_TLCC(feature_subset, feature_selected), use_container_width=True)
