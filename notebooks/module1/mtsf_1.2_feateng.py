# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture 1.2: Features Engineering & Recipes -------------------------------

# Goals:
# - Learn advanced features engineering workflows & techniques
# - Learn how to use pytimetk
# - Learn how to create Scikit-Learn preprocessing pipelines

# Challenges:
# - Challenge 1 - Feature Engineering



# Packages ----------------------------------------------------------------

import sys
sys.path.insert(0, 'src/Python/utils')
from utils import (
    load_data, log_interval, inv_log_interval, standardize, inv_standardize, 
    plot_time_series_regression, plot_acf_pacf 
)
import polars as pl
import pytimetk as tk
import re
import pickle



# Data --------------------------------------------------------------------

email_df = load_data('data/email/', 'email_prep', ext = '.parquet')
email_df.glimpse()



# Features Engineering ----------------------------------------------------

# Pre-processing Data

email_df.tk.plot_timeseries('ds', 'y', smooth = False)

df_names = email_df.columns

# filter out the first part of the data with zeros
data_prep_df = email_df \
    .tk.filter_by_time(date_column = 'ds', start_date = '2018-07-03')

y_lb = 0
y_ub = data_prep_df['y'].max() * 1.10
y_offset = 1

data_prep_df = data_prep_df \
    .with_columns(log_interval(lb = y_lb, ub = y_ub, offset = y_offset).alias('y'))

y_mean = data_prep_df['y'].mean()
y_std = data_prep_df['y'].std()

data_prep_df = data_prep_df \
    .with_columns(standardize(mean = y_mean, stdev = y_std).alias('y'))

data_prep_df = data_prep_df \
    .tk.anomalize(
        date_column = 'ds', value_column = 'y', method = 'stl', 
        iqr_alpha = 0.02, max_anomalies = 0.2, clean_alpha = 0.5, 
        bind_data = True
    ) \
    .with_columns(
        pl.when(pl.col('anomaly') == 'Yes')
        .then(pl.col('observed_clean'))
        .otherwise(pl.col('y'))
        .alias('y')
    ) \
    .select(df_names)
    
data_prep_df = data_prep_df \
    .with_columns(
        pl.col('pageViews').log1p().alias('pageViews'),
        pl.col('organicSearches').log1p().alias('organicSearches'),
        pl.col('sessions').log1p().alias('sessions')
    ) \
    .with_columns(
        standardize('pageViews').alias('pageViews'),
        standardize('organicSearches').alias('organicSearches'),
        standardize('sessions').alias('sessions')
    )

data_prep_df.tk.plot_timeseries('ds', 'y', smooth = False)
data_prep_df \
    .drop('unique_id') \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)


# * Time-Based Features ---------------------------------------------------

# calendar
data_prep_df \
    .tk.augment_timeseries_signature(date_column = 'ds') \
    .glimpse()

# holidays
data_prep_df \
    .tk.augment_holiday_signature(date_column = 'ds', country_name = 'US') \
    .glimpse()


# * Trend-Based Features --------------------------------------------------

# linear trend
data_prep_df \
    .tk.augment_timeseries_signature(date_column = 'ds') \
    .select('unique_id', 'ds', 'y', 'ds_index_num') \
    .with_columns(standardize('ds_index_num').alias('ds_index_num')) \
    .melt(id_vars = ['unique_id', 'ds'], value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)

data_prep_df \
    .tk.augment_timeseries_signature(date_column = 'ds') \
    .select('unique_id', 'ds', 'y', 'ds_index_num') \
    .with_columns(standardize('ds_index_num').alias('ds_index_num')) \
    .pipe(plot_time_series_regression)

# nonlinear trend - basis splines
data_prep_df \
    .tk.augment_spline(
        date_column = 'ds', value_column = 'y', 
        spline_type = 'bs', df = 4, degree = 2, knots = [0.25, 0.5]
    ) \
    .drop('pageViews', 'organicSearches', 'sessions', 'promo') \
    .melt(id_vars = ['unique_id', 'ds'], value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)


# * Seasonal Features -----------------------------------------------------

# - seasonal dummy variables with one-hot-encoding

# weekly seasonality 
data_prep_df \
    .tk.augment_timeseries_signature(date_column = 'ds') \
    .select('unique_id', 'ds', 'y', 'ds_wday_lbl') \
    .to_dummies(columns = ['ds_wday_lbl']) \
    .pipe(plot_time_series_regression)

# monthly seasonality
data_prep_df \
    .tk.augment_timeseries_signature(date_column = 'ds') \
    .select('unique_id', 'ds', 'y', 'ds_month_lbl') \
    .to_dummies(columns = ['ds_month_lbl']) \
    .pipe(plot_time_series_regression)


# * Interaction Features --------------------------------------------------

# day of week * week 2 of the month
data_prep_df \
    .tk.augment_timeseries_signature(date_column = 'ds') \
    .select('unique_id', 'ds', 'y', 'ds_mweek', 'ds_wday') \
    .to_dummies(columns = ['ds_mweek']) \
    .with_columns(
        (pl.col('ds_wday') * pl.col('ds_mweek_2')).alias('wday_times_week2')
    ) \
    .select('unique_id', 'ds', 'y', 'ds_wday', 'ds_mweek_2', 'wday_times_week2') \
    .pipe(plot_time_series_regression)


# * Rolling Average Features ----------------------------------------------

data_prep_df \
    .tk.augment_rolling(
        date_column = 'ds', value_column = 'y',
        window_func = 'mean', window = [7, 14, 30, 90]
    ) \
    .drop('pageViews', 'organicSearches', 'sessions', 'promo') \
    .drop_nulls() \
    .pipe(plot_time_series_regression)
    

# Exponential Weighted Rolling Average
data_prep_df \
    .tk.augment_ewm(
        date_column = 'ds', value_column = 'y', window_func = 'mean', alpha = 0.1
    ) \
    .tk.augment_ewm(
        date_column = 'ds', value_column = 'y', window_func = 'mean', alpha = 0.25
    ) \
    .tk.augment_ewm(
        date_column = 'ds', value_column = 'y', window_func = 'mean', alpha = 0.5
    ) \
    .drop('pageViews', 'organicSearches', 'sessions', 'promo') \
    .pipe(plot_time_series_regression)


# * Lag Features ----------------------------------------------------------

data_prep_df.pipe(plot_acf_pacf, column = 'y', lags = 120)

data_prep_df \
    .tk.augment_lags(date_column = 'ds', value_column = 'y', lags = [1, 7, 14, 30, 45]) \
    .drop('pageViews', 'organicSearches', 'sessions', 'promo') \
    .drop_nulls() \
    .pipe(plot_time_series_regression)


# * Fourier Series Features -----------------------------------------------

data_prep_df \
    .tk.augment_fourier(date_column = 'ds', periods = [1, 7, 14, 30, 45], max_order = 2) \
    .drop('pageViews', 'organicSearches', 'sessions', 'promo') \
    .pipe(plot_time_series_regression)

# Wavelet Series Features
data_prep_df \
    .tk.augment_wavelet(
        date_column = 'ds', value_column = 'y', scales = [7, 14, 30, 90],  
        sample_rate = 7, method = 'bump'
    ) \
    .drop('pageViews', 'organicSearches', 'sessions', 'promo') \
    .pipe(plot_time_series_regression)


# * External Regressor Features -------------------------------------------

# Promo data features 
data_prep_df \
    .drop('pageViews', 'organicSearches', 'sessions') \
    .pipe(plot_time_series_regression)

# Analytics data features
data_prep_df \
    .tk.augment_lags(
        date_column = 'ds',
        value_column = ['pageViews', 'organicSearches', 'sessions'], 
        lags = [7, 30]
    ) \
    .drop_nulls() \
    .drop('pageViews', 'organicSearches', 'sessions', 'promo') \
    .pipe(plot_time_series_regression)



# Features Engineering Workflow -------------------------------------------

# * Pre-processing Data ---------------------------------------------------

# - Filter to keep only useful period
# - Apply log_interval transform
# - Apply standardization transform
# - Clean anomalies

# filter out the first part of the data with zeros
data_prep_df = email_df \
    .tk.filter_by_time(date_column = 'ds', start_date = '2018-07-03')

y_lb = 0
y_ub = data_prep_df['y'].max() * 1.10
y_offset = 1

data_prep_df = data_prep_df \
    .with_columns(log_interval(lb = y_lb, ub = y_ub, offset = y_offset).alias('y'))

y_mean = data_prep_df['y'].mean()
y_std = data_prep_df['y'].std()

data_prep_df = data_prep_df \
    .with_columns(standardize(mean = y_mean, stdev = y_std).alias('y'))

df_names = data_prep_df.columns
data_prep_df = data_prep_df \
    .tk.anomalize(
        date_column = 'ds', value_column = 'y', method = 'stl', 
        iqr_alpha = 0.02, max_anomalies = 0.2, clean_alpha = 0.5, 
        bind_data = True
    ) \
    .with_columns(
        pl.when(pl.col('anomaly') == 'Yes')
        .then(pl.col('observed_clean'))
        .otherwise(pl.col('y'))
        .alias('y')
    ) \
    .select(df_names)
    
data_prep_df = data_prep_df.drop('pageViews', 'organicSearches', 'sessions')

# Store transformation parameters
params = {
    'lower_bound': y_lb,
    'upper_bound': y_ub,
    'offset': y_offset,
    'mean_y': y_mean,
    'stdev_y': y_std
}


# * Creating Features -----------------------------------------------------

# - Extend to Future Window
# - Add any lags to full dataset
# - Add rolling features to full dataset
# - Add fourier terms to full dataset
# - Add wavelet terms to full dataset
# - Add calendar variable to full dataset (+ one-hot-encoding)
# - Add any external regressors to full dataset (keep Promo)

horizon = 7 * 8 # 8 weeks
lag_periods = [7 * 8]
rolling_periods = [30, 60, 90]

data_prep_full_df = data_prep_df \
    .tk.future_frame(date_column = 'ds', length_out = horizon) \
    .tk.augment_lags(date_column = 'ds', value_column = 'y', lags = lag_periods) \
    .tk.augment_rolling(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', window = rolling_periods
    ) \
    .tk.augment_fourier(
        date_column = 'ds', periods = [7, 14, 30, 90, 365], max_order = 2
    ) \
    .tk.augment_wavelet(
        date_column = 'ds', value_column = 'y', scales = [7, 14, 30, 90],  
        sample_rate = 7, method = 'bump'
    ) \
    .tk.augment_timeseries_signature(date_column = 'ds') \
    .drop(
        'ds_year_iso', 'ds_yearstart', 'ds_yearend', 'ds_leapyear',
        'ds_quarteryear', 'ds_quarterstart', 'ds_quarterend', 
        'ds_monthstart', 'ds_monthend', 'ds_qday', 'ds_yday',
        'ds_hour', 'ds_minute', 'ds_second', 'ds_msecond', 'ds_nsecond', 'ds_am_pm'
    ) \
    .with_columns(standardize('ds_index_num').alias('ds_index_num')) \
    .to_dummies(columns = ['ds_wday_lbl', 'ds_month_lbl']) \
    .drop('ds_wday_lbl_Monday', 'ds_month_lbl_January') \
    .with_columns(
        pl.when(pl.col('promo').is_null())
        .then(0)
        .otherwise(pl.col('promo'))
        .alias('promo')
    )

data_prep_full_df.glimpse()


# * Separate into Modelling & Forecast Data -------------------------------

data_model_df = data_prep_full_df.head(n = -horizon)
forecast_df = data_prep_full_df.tail(n = horizon)


# * Create different Features Sets ('Recipes') ----------------------------

# base feature set (no lags, no wavelets)
r = re.compile(r'(lag)|(bump)')
base_fs = [i for i in data_model_df.columns if not r.search(i)]
base_fs

# wavalets feature set (no lags)
r = re.compile(r'lag')
wave_fs = [i for i in data_model_df.columns if not r.search(i)]
wave_fs

# lag feature set (no wavelets)
r = re.compile(r'bump')
lag_fs = [i for i in data_model_df.columns if not r.search(i)]
lag_fs

feature_sets = {
    'base': base_fs,
    'wave': wave_fs,
    'lag': lag_fs
}


# * Save Artifacts --------------------------------------------------------

feature_engineering_artifacts = {
    'data_prep_df': data_model_df,
    'forecast_df': forecast_df,
    'transform_params': params, 
    'feature_sets': feature_sets
}
feature_engineering_artifacts

# Serialize the object to a binary format
with open('data/email/artifacts/feature_engineering_artifacts_list.pkl', 'wb') as file:
    pickle.dump(feature_engineering_artifacts, file)



# Testing - Modelling Workflow --------------------------------------------

from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse

# Nixtla's workflow
# 1. set the model engine (usually it contains preprocessing and 
#    feature engineering information)
# 2. evaluate the model against a test set (it is done by cross-validation)
# 3. re-fit the model on the whole data
# 4. produce forecast out-of-sample

with open('data/email/artifacts/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
feature_sets = data_loaded['feature_sets']
params = data_loaded['transform_params']
horizon = 7 * 8 # 8 weeks


# * Features Sets (Recipes) -----------------------------------------------

# in our case we have created manually all the features, hence we need
# to create a different dataset for each feature set that we want to test 
data_base = data_prep_df.select(feature_sets['base'])
data_base.glimpse()


# * Model Engine Specification --------------------------------------------

# Linear Regression
fcst = MLForecast(models = LinearRegression(), freq = '1d')


# * Evaluation ------------------------------------------------------------

cv_result = fcst \
    .cross_validation(data_base, n_windows = 1, h = horizon, static_features = [])
cv_result

# Plot Forecasts
plot_series(forecasts_df = cv_result.drop('cutoff'), engine = 'plotly').show()

# Accuracy
accuracy_result = evaluate(
    df = cv_result.drop('cutoff'), train_df = data_prep_df,
    metrics = [bias, mae, mape, mse, rmse], agg_fn = 'mean'
)
accuracy_result


# * Model Re-Fitting ---------------------------------------------------------

fcst.fit(data_base, static_features = [])


# * Forecasting -----------------------------------------------------------

preds = fcst.predict(h = horizon, X_df = forecast_df)
preds

plot_series(df = data_prep_df, forecasts_df = preds, engine = 'plotly').show()


# * Back-transform --------------------------------------------------------

data_back_df = data_base \
    .select('unique_id', 'ds', 'y') \
    .with_columns(
        inv_standardize(mean = params['mean_y'], stdev = params['stdev_y']).alias('y')
    ) \
    .with_columns(
        inv_log_interval(
            lb = params['lower_bound'], ub = params['upper_bound'], offset = params['offset']
        ).alias('y')
    )

preds_back_df = preds \
    .with_columns(
        inv_standardize('LinearRegression', mean = params['mean_y'], stdev = params['stdev_y']).alias('LinearRegression')
    ) \
    .with_columns(
        inv_log_interval(
            'LinearRegression', lb = params['lower_bound'], ub = params['upper_bound'], offset = params['offset']
        ).alias('LinearRegression')
    )

plot_series(df = data_back_df, forecasts_df = preds_back_df, engine = 'plotly').show()

