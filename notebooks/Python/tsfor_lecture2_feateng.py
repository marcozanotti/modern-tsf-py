# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 2: Features Engineering & Recipes -------------------------------
# Marco Zanotti

# Goals:
# - Learn advanced features engineering workflows & techniques
# - Learn how to use pytimetk
# - Learn how to create Scikit-Learn preprocessing pipelines

# Challenges:
# - Challenge 1 - Feature Engineering



# Packages ----------------------------------------------------------------

import numpy as np
import pandas as pd
# import polars as pl
import re
import pickle

import pytimetk as tk

from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mase, mse, rmae, rmse

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import python_extensions as pex

pd.set_option("display.max_rows", 3)



# Data --------------------------------------------------------------------

# Pandas
subscribers_df = pd.read_csv(
    "data/subscribers/subscribers.csv", 
    parse_dates=['optin_time']
)
analytics_df = pd.read_csv(
    "data/subscribers/analytics_hourly.csv", 
    parse_dates=['dateHour'],
    date_format='%Y%m%d%H'
)
events_df = pd.read_csv(
    "data/subscribers/events.csv", parse_dates=['event_date']
)

subscribers_df.glimpse()
analytics_df.glimpse()
events_df.glimpse()



# Features Engineering ----------------------------------------------------

# Pre-processing Data

# transform time series with log1p and standardization
# fix missing values at beginning of series
# clean anomaly on '2018-11-19'

# subscribers data
subscribers_prep_df = subscribers_df \
    .assign(value = 1) \
    .rename(columns = {"optin_time": "date"}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0) \
    .transform_columns(columns = 'value', transform_func = pex.log_interval) \
    .transform_columns(columns = 'value', transform_func = pex.standardize) \
    .filter_by_time(
        date_column = 'date', 
        start_date = '2018-07-03' 
    ) \
    .anomalize(
        date_column = 'date', value_column = 'value',
        method = 'twitter',
        iqr_alpha = 0.025,
        max_anomalies = 0.2,
        clean_alpha = 0.5,
        clean = 'min-max'
    ) \
    .assign(id = 'subscribers') \
    .reindex(columns = ['id', 'date', 'observed_clean']) \
    .rename(columns = {'observed_clean': 'value'})   
subscribers_prep_df \
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        smooth = False
    )

# Nixtla's default format 
# 'unique_id', 'ds', 'y'
subscribers_prep_df = subscribers_prep_df \
    .rename(
        columns = {
            'id': 'unique_id',
            'date': 'ds',
            'value': 'y'
        }
    )
subscribers_prep_df


# analytics data
analytics_prep_df = analytics_df \
    .rename(columns = {'dateHour': 'date'}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = ['pageViews', 'organicSearches', 'sessions'],
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D'
    ) \
    .fillna(0) \
    .transform_columns(
        columns = ['pageViews', 'organicSearches', 'sessions'], 
        transform_func = np.log1p
    ) \
    .transform_columns(
        columns = ['pageViews', 'organicSearches', 'sessions'], 
        transform_func = pex.standardize
    ) \
    .filter_by_time(
        date_column = 'date', 
        start_date = '2018-07-03' 
    )
analytics_prep_df \
    .melt(id_vars = 'date', value_vars = ['pageViews', 'organicSearches', 'sessions']) \
    .groupby('variable') \
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        color_column = 'variable',
        facet_nrow = 3,
        smooth = False
    )

# events data
events_prep_df = events_df \
    .assign(event = 1) \
    .rename(columns = {"event_date": "date"}) \
    .drop("id", axis = 1) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'event',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D'
    ) \
    .fillna(0)
events_prep_df['event'] = events_prep_df['event'].astype('int64')  
events_prep_df \
    .plot_timeseries(
        date_column = 'date',
        value_column = 'event',
        smooth = False
    )


# * Time-Based Features ---------------------------------------------------

# calendar
subscribers_prep_df \
    .augment_timeseries_signature(date_column = 'ds')

# holidays
subscribers_prep_df \
    .augment_holiday_signature(date_column = 'ds', country_name = 'US')


# * Trend-Based Features --------------------------------------------------

# linear trend
subscribers_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_index_num']) \
    .plot_time_series_regression()

# nonlinear trend - basis splines
subscribers_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_index_num']) \
    .augment_bsplines(column_name = 'ds_index_num', df = 8, degree = 3) \
    .drop('ds_index_num', axis = 1) \
    .plot_time_series_regression()


# * Seasonal Features -----------------------------------------------------

# weekly seasonality (one-hot-encoding)
data_prep_tmp = subscribers_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_wday_lbl'])
data_prep_tmp = pd.get_dummies(
    data_prep_tmp, columns = ['ds_wday_lbl'], 
    drop_first = True, dtype = int
)
data_prep_tmp \
    .plot_time_series_regression()

# monthly seasonality
data_prep_tmp = subscribers_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_month_lbl'])
data_prep_tmp = pd.get_dummies(
    data_prep_tmp, columns = ['ds_month_lbl'], 
    drop_first = True, dtype = int
)
data_prep_tmp \
    .plot_time_series_regression()


# * Interaction Features --------------------------------------------------

# day of week * week 2 of the month
data_prep_tmp = subscribers_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_mweek', 'ds_wday'])
data_prep_tmp = pd.get_dummies(
    data_prep_tmp, columns = ['ds_mweek'], 
    drop_first = True, dtype = int
) \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_wday', 'ds_mweek_2']) \
    .assign(wday_times_week2 = lambda x: x['ds_wday'] * x['ds_mweek_2'])
data_prep_tmp \
    .plot_time_series_regression()


# * Rolling Average Features ----------------------------------------------

subscribers_prep_df \
    .augment_rolling(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        window = [7, 14, 30, 90]
    ) \
    .dropna(axis = 0) \
    .plot_time_series_regression()

# Exponential Weighted Moving Average
subscribers_prep_df \
    .augment_ewm(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        alpha = 0.1
    ) \
    .augment_ewm(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        alpha = 0.2
    ) \
    .augment_ewm(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        alpha = 0.3
    ) \
    .plot_time_series_regression()


# * Lag Features ----------------------------------------------------------

plot_acf(subscribers_prep_df["y"], lags = 50)
plot_pacf(subscribers_prep_df["y"], lags = 50)

subscribers_prep_df \
    .augment_lags(
        date_column = 'ds', 
        value_column = 'y', 
        lags = [1, 7, 14, 30, 90]
    ) \
    .dropna(axis = 0) \
    .plot_time_series_regression()


# * Fourier Series Features -----------------------------------------------

subscribers_prep_df \
    .augment_fourier(
        date_column = 'ds', 
        periods = [1, 7, 14, 30, 90], 
        max_order = 2
    ) \
    .plot_time_series_regression()


# * Wavelet Series Features -----------------------------------------------

tk.augment_wavelet(
    subscribers_prep_df, 
    date_column = 'ds', 
    value_column = 'y',
    scales = [7, 14, 30, 90],  
    sample_rate = 7,
    method = 'bump'
) \
    .plot_time_series_regression()


# * External Regressor Features -------------------------------------------

# Event data features 
subscribers_prep_df \
    .merge(events_prep_df, left_on = 'ds', right_on = 'date', how = 'left') \
    .drop('date', axis = 1) \
    .fillna(0) \
    .plot_time_series_regression()

# Analytics data features
subscribers_prep_df \
    .merge(analytics_prep_df, left_on = 'ds', right_on = 'date', how = 'left') \
    .drop('date', axis = 1) \
    .augment_lags(
        date_column = 'ds',
        value_column = ['pageViews', 'organicSearches', 'sessions'], 
        lags = [7, 42]
    ) \
    .dropna() \
    .plot_time_series_regression()



# Features Engineering Workflow -------------------------------------------

# * Pre-processing Data ---------------------------------------------------

# - Aggregate to daily frequency
# - Pad with zero values
# - Apply log1p transform
# - Apply standardization transform
# - Filter to kee only useful period
# - Clean anomalies

# subscribers data
subscribers_prep_df = subscribers_df \
    .assign(value = 1) \
    .rename(columns = {"optin_time": "date"}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0) \
    .transform_columns(columns = 'value', transform_func = pex.log_interval) \
    .transform_columns(columns = 'value', transform_func = pex.standardize) \
    .filter_by_time(
        date_column = 'date', 
        start_date = '2018-07-03' 
    ) \
    .anomalize(
        date_column = 'date', value_column = 'value',
        method = 'twitter',
        iqr_alpha = 0.025,
        max_anomalies = 0.2,
        clean_alpha = 0.5,
        clean = 'min-max'
    ) \
    .assign(id = 'subscribers') \
    .reindex(columns = ['id', 'date', 'observed_clean']) \
    .rename(columns = {'observed_clean': 'value'})

# Store transformation parameters
params = {
    'lower_bound': 0,
    'upper_bound': 3649.8,
    'offset': 1,
    'mean_x': -5.82796116564194,
    'stdev_x': 1.533985803607424
}

# Nixtla's default format 
# 'unique_id', 'ds', 'y'
subscribers_prep_df = subscribers_prep_df \
    .rename(
        columns = {
            'id': 'unique_id',
            'date': 'ds',
            'value': 'y'
        }
    )
subscribers_prep_df

# events data
events_prep_df = events_df \
    .assign(event = 1) \
    .rename(columns = {"event_date": "date"}) \
    .drop("id", axis = 1) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'event',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D'
    ) \
    .fillna(0)
events_prep_df['event'] = events_prep_df['event'].astype('int64')  
events_prep_df


# * Creating Features -----------------------------------------------------

# - Extend to Future Window
# - Add any lags to full dataset
# - Add rolling features to full dataset
# - Add fourier terms to full dataset
# - Add calendar variable to full dataset (+ one-hot-encoding)
# - Add holidays variables to full dataset (+ one-hot-encoding)
# - Add splines
# - Add any external regressors to full dataset
# - Add any interaction terms to full dataset

horizon = 7 * 8 # 8 weeks
lag_periods = [7 * 8]
rolling_periods = [30, 60, 90]

data_prep_full_df = subscribers_prep_df \
    .future_frame(date_column = 'ds', length_out = horizon) \
    .augment_lags(
        date_column = 'ds', value_column = 'y', 
        lags = lag_periods
    ) \
    .augment_rolling(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', window = rolling_periods
    ) \
    .augment_ewm(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', alpha = 0.1
    ) \
    .augment_ewm(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', alpha = 0.2
    ) \
    .augment_ewm(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', alpha = 0.3
    ) \
    .augment_fourier(
        date_column = 'ds', 
        periods = [7, 14, 30, 90, 365], 
        max_order = 2
    ) \
    .augment_timeseries_signature(date_column = 'ds') \
    .drop(
        [
            'ds_year_iso', 'ds_yearstart', 'ds_yearend', 'ds_leapyear',
            'ds_quarteryear', 'ds_quarterstart', 'ds_quarterend', 
            'ds_monthstart', 'ds_monthend', 'ds_qday', 'ds_yday',
            'ds_hour', 'ds_minute', 'ds_second', 'ds_msecond', 'ds_nsecond', 'ds_am_pm'
        ], 
        axis = 1
    ) \
    .augment_holiday_signature(date_column = 'ds', country_name = 'US') \
    .drop('holiday_name', axis = 1) \
    .augment_bsplines(column_name = 'ds_index_num', df = 8, degree = 3) \
    .merge(events_prep_df, left_on = 'ds', right_on = 'date', how = 'left') \
    .drop('date', axis = 1) \
    .fillna({'event': 0})

# create interactions and combine
interactions_df = pd.get_dummies(
    data_prep_full_df, columns = ['ds_mweek'], 
    drop_first = True, dtype = int
) \
    .assign(inter_wday_week2 = lambda x: x['ds_wday'] * x['ds_mweek_2']) \
    .reindex(columns = ['inter_wday_week2', 'ds_mweek_2'])
data_prep_full_df = pd.concat([data_prep_full_df, interactions_df], axis = 1)

# one-hot-encoding of dummy variables
data_prep_full_dummy_df = pd.get_dummies(
    data_prep_full_df, columns = ['ds_wday_lbl', 'ds_month_lbl'], 
    drop_first = True, dtype = int
)
data_prep_full_dummy_df


# * Separate into Modelling & Forecast Data -------------------------------

data_prep_df = data_prep_full_dummy_df.head(n = -horizon)
forecast_df = data_prep_full_dummy_df.tail(n = horizon)


# * Create different Features Sets ('Recipes') ----------------------------

# base feature set (no lags, no splines)
r = re.compile(r'(lag)|(spline)')
base_fs = [i for i in data_prep_df.columns if not r.search(i)]
base_fs

# spline feature set (no lags)
r = re.compile(r'lag')
spline_fs = [i for i in data_prep_df.columns if not r.search(i)]
spline_fs

# lag feature set (no spline)
r = re.compile(r'spline')
lag_fs = [i for i in data_prep_df.columns if not r.search(i)]
lag_fs

feature_sets = {
    'base': base_fs,
    'spline': spline_fs,
    'lag': lag_fs
}


# * Save Artifacts --------------------------------------------------------

feature_engineering_artifacts = {
    'data_prep_df': data_prep_df,
    'forecast_df': forecast_df,
    'transform_params': params, 
    'feature_sets': feature_sets
}
feature_engineering_artifacts

# Serialize the object to a binary format
with open('artifacts/Python/feature_engineering_artifacts_list.pkl', 'wb') as file:
    pickle.dump(feature_engineering_artifacts, file)



# Testing - Modelling Workflow --------------------------------------------

# Nixtla's workflow
# 1. set the model engine (usually it contains preprocessing and 
#    feature engineering information)
# 2. evaluate the model against a test set (it is done by cross-validation)
# 3. re-fit the model on the whole data
# 4. produce forecast out-of-sample

with open('artifacts/Python/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
feature_sets = data_loaded['feature_sets']
params = data_loaded['transform_params']
horizon = 7 * 8 # 8 weeks


# * Features Sets (Recipes) -----------------------------------------------

# in our case we have created manually all the features, hence we need
# to create a different dataset for each feature set that we want to test 
data_base = data_prep_df.reindex(columns = feature_sets['base'])
data_spline = data_prep_df.reindex(columns = feature_sets['spline'])
data_lag = data_prep_df \
    .reindex(columns = feature_sets['lag']) \
    .dropna()


# * Model Engine Specification --------------------------------------------

# Linear Regression
fcst_base = MLForecast(models = LinearRegression(), freq = 'D')
fcst_spline = MLForecast(models = LinearRegression(), freq = 'D')
fcst_lag = MLForecast(models = LinearRegression(), freq = 'D')


# * Evaluation ------------------------------------------------------------

cv_result_base = fcst_base.cross_validation(
    data_base, 
    n_windows = 1,
    h = horizon, 
    static_features = []
) \
    .rename(columns = {'LinearRegression': 'linreg_base'})
cv_result_base

cv_result_spline = fcst_spline.cross_validation(
    data_spline, 
    n_windows = 1,
    h = horizon, 
    static_features = []
) \
    .rename(columns = {'LinearRegression': 'linreg_spline'})
cv_result_lag = fcst_lag.cross_validation(
    data_lag, 
    n_windows = 1,
    h = horizon, 
    static_features = []
) \
    .rename(columns = {'LinearRegression': 'linreg_lag'})

cv_result = cv_result_base \
    .merge(cv_result_spline, on = ['unique_id', 'ds', 'y', 'cutoff'], how = 'left') \
    .merge(cv_result_lag, on = ['unique_id', 'ds', 'y', 'cutoff'], how = 'left')

# Plot Forecasts
plot_series(
    forecasts_df = cv_result.drop('cutoff', axis = 1), 
    engine = 'plotly'
).show()

# Accuracy
accuracy_result = evaluate(
    df = cv_result.drop(columns = 'cutoff'),
    train_df = data_prep_df,
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)
accuracy_result


# * Model Re-Fitting ---------------------------------------------------------

fcst_base.fit(data_base, static_features = [])
fcst_spline.fit(data_spline, static_features = [])
fcst_lag.fit(data_lag, static_features = [])


# * Forecasting -----------------------------------------------------------

preds_base = fcst_base.predict(h = horizon, X_df = forecast_df) \
    .rename(columns = {'LinearRegression': 'linreg_base'})
preds_base

preds_spline = fcst_spline.predict(h = horizon, X_df = forecast_df) \
    .rename(columns = {'LinearRegression': 'linreg_spline'})
preds_lag = fcst_lag.predict(h = horizon, X_df = forecast_df) \
    .rename(columns = {'LinearRegression': 'linreg_lag'})

preds = preds_base \
    .merge(preds_spline, on = ['unique_id', 'ds'], how = 'left') \
    .merge(preds_lag, on = ['unique_id', 'ds'], how = 'left')

plot_series(
    df = data_prep_df, 
    forecasts_df = preds, 
    engine = 'plotly'
).show()


# * Back-transform --------------------------------------------------------

data_back_df = data_prep_df \
    .transform_columns(
        columns = 'y', 
        transform_func = lambda x: pex.inv_standardize(x, params['mean_x'], params['stdev_x'])
    ) \
    .transform_columns(
        columns = 'y', 
        transform_func = lambda x: pex.inv_log_interval(
            x, params['lower_bound'], params['upper_bound'], params['offset']
        )
    )

preds_back_df = preds \
    .transform_columns(
        columns = ['linreg_base', 'linreg_spline', 'linreg_lag'], 
        transform_func = lambda x: pex.inv_standardize(x, params['mean_x'], params['stdev_x'])
    ) \
    .transform_columns(
        columns = ['linreg_base', 'linreg_spline', 'linreg_lag'], 
        transform_func = lambda x: pex.inv_log_interval(
            x, params['lower_bound'], params['upper_bound'], params['offset']
        )
    )

plot_series(
    df = data_back_df, 
    forecasts_df = preds_back_df, 
    engine = 'plotly'
).show()

