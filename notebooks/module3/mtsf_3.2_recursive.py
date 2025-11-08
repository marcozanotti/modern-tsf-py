# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture 3.2: Recursive Time Series Algorithms ----------------------------

# Goals:
# - Panel Recursivity



# Packages ----------------------------------------------------------------

import sys
sys.path.insert(0, 'src/Python/utils')
from utils import (
    load_data, calibrate_evaluate_plot, print_accuracy_table,
    get_best_model_name, get_best_model_forecast
)
import polars as pl
import pytimetk as tk

from functools import partial
from utilsforecast.feature_engineering import (
    fourier, trend, time_features, pipeline
)
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from utilsforecast.plotting import plot_series
from mlforecast.lag_transforms import (
    RollingMean, ExpandingMean, ExponentiallyWeightedMean
)

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



# Data --------------------------------------------------------------------

m4_df = load_data('data/m4/', 'm4_prep_sample', ext = '.parquet') \
    .with_columns(pl.col('ds').dt.cast_time_unit('ns').dt.replace_time_zone(None))
m4_df.glimpse()

m4_df \
    .group_by('unique_id') \
    .tk.plot_timeseries('ds', 'y', facet_ncol = 2, smooth = False)

m4_df.group_by('unique_id').count()


# * Forecast Horizon ------------------------------------------------------

horizon = 24 * 2 # 2 days


# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
intervals = PredictionIntervals(h = horizon, n_windows = 2)


# * Feature Engineering ---------------------------------------------------

# We use Nixtla's direct feature engineering utilities to create 
# time based features (calendar, trend, and fourier) and lags.

# custom feature function
def is_weekend(times):
    dow = times.dt.weekday()
    return dow >= 6

features = [
    trend,
    partial(fourier, season_length = 12, k = 1),
    partial(fourier, season_length = 24, k = 1),
    partial(fourier, season_length = 36, k = 1),
    partial(fourier, season_length = 48, k = 1),
    partial(time_features, features = ['hour', 'day', 'weekday', is_weekend]),
]

# also separate into modelling & forecast datasets
data_model_df, forecast_df = pipeline(
    m4_df, features = features, freq = '1h', h = horizon, 
)

data_model_df.glimpse()
forecast_df.glimpse()



# Machine Learning Models -------------------------------------------------


# * Global Non-Recursive --------------------------------------------------

models_ml = [
    LinearRegression(),
    ElasticNet(l1_ratio = 0.5, alpha = 0.01),
    RandomForestRegressor(
        n_estimators = 100,
        criterion = 'squared_error',
        max_depth = None,
        min_samples_split = 2,
        max_features = 'sqrt',
        random_state = 0
    ), 
    XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'reg:squarederror',
        random_state = 0
    )
]

mlf_no_rec = MLForecast(
    models = models_ml, 
    freq = '1h', 
    num_threads = -1,
    lags = [horizon],
    lag_transforms = {
        horizon: [
            ExpandingMean(),
            RollingMean(window_size = 12),
            RollingMean(window_size = 24),
            RollingMean(window_size = 36),
            RollingMean(window_size = 48),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

mlf_no_rec.preprocess(data_model_df, static_features = [], dropna = False)


# * Global Recursive ------------------------------------------------------

models_ml_rec = {
    'LinearRegression_rec': LinearRegression(),
    'ElasticNet_rec': ElasticNet(
        l1_ratio = 0.5, 
        alpha = 0.01
    ),
    'RandomForestRegressor_rec': RandomForestRegressor(
        n_estimators = 100,
        criterion = 'squared_error',
        max_depth = None,
        min_samples_split = 2,
        max_features = 'sqrt',
        random_state = 0
    ), 
    'XGBRegressor_rec': XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'reg:squarederror',
        random_state = 0
    )
}

mlf_rec = MLForecast(
    models = models_ml_rec, 
    freq = '1h', 
    num_threads = -1,
    lags = [1, 2, 6, 12, 24],
    lag_transforms = {
        1: [
            ExpandingMean(), 
            ExponentiallyWeightedMean(alpha = 0.3),
            RollingMean(window_size = 12)
        ],
        6: [
            RollingMean(window_size = 12, min_samples = 1)
        ],
        24: [
            RollingMean(window_size = 24, min_samples = 1),
            ExponentiallyWeightedMean(alpha = 0.1)
        ]
    }
)

mlf_rec.preprocess(data_model_df, static_features = [], dropna = False)


# * Evaluation ------------------------------------------------------------

# ** Non-Recursive --------------------------------------------------------

cv_res_no_rec = calibrate_evaluate_plot(
    mlf_no_rec, df = data_model_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3, by_id = True  
)

cv_res_no_rec['cv_results']

# Local accuracy
cv_res_no_rec_local = cv_res_no_rec['accuracy_table']
cv_res_no_rec_local

# Gloabl accuracy
cv_res_no_rec_global = cv_res_no_rec['accuracy_table'] \
        .group_by('metric') \
        .agg(pl.col(pl.NUMERIC_DTYPES).mean())
print_accuracy_table(cv_res_no_rec_global)

cv_res_no_rec['plot'].show()

# ** Recursive ------------------------------------------------------------

cv_res_rec = calibrate_evaluate_plot(
    mlf_rec, df = data_model_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3, by_id = True  
)

cv_res_rec['cv_results']

# Local accuracy
cv_res_rec_local = cv_res_rec['accuracy_table']
cv_res_rec_local

# Gloabl accuracy
cv_res_rec_global = cv_res_rec['accuracy_table'] \
        .group_by('metric') \
        .agg(pl.col(pl.NUMERIC_DTYPES).mean())
print_accuracy_table(cv_res_rec_global)

cv_res_rec['plot'].show()

# ** Combining accuracy ---------------------------------------------------

cv_res_accuracy_local = cv_res_no_rec_local \
    .join(cv_res_rec_local, on = ['unique_id', 'metric'])
cols = ['unique_id', 'metric'] + sorted([c for c in cv_res_accuracy_local.columns if c not in ['unique_id', 'metric']])
cv_res_accuracy_local = cv_res_accuracy_local.select(cols)

cv_res_accuracy_global = cv_res_no_rec_global \
    .join(cv_res_rec_global, on = 'metric')
cols = ['metric'] + sorted([c for c in cv_res_accuracy_global.columns if c != 'metric'])
cv_res_accuracy_global = cv_res_accuracy_global.select(cols)
print_accuracy_table(cv_res_accuracy_global)


# * Refitting & Forecasting -----------------------------------------------

# ** Non-Recursive --------------------------------------------------------

fit_no_rec = mlf_no_rec.fit(df = data_model_df, prediction_intervals = intervals, static_features = [])
preds_df_no_rec = fit_no_rec.predict(h = horizon, level = levels, X_df = forecast_df)

# forecast with best model locally
get_best_model_name(cv_res_no_rec_local, 'rmse', by_id = True)
preds_best_df_no_rec_local = get_best_model_forecast(preds_df_no_rec, cv_res_no_rec_local, 'rmse', by_id = True)

# forecast with best model globally
get_best_model_name(cv_res_no_rec_global, 'rmse')
preds_best_df_no_rec_global = get_best_model_forecast(preds_df_no_rec, cv_res_no_rec_global, 'rmse')

# ** Recursive ------------------------------------------------------------

fit_rec = mlf_rec.fit(df = data_model_df, prediction_intervals = intervals, static_features = [])
preds_df_rec = fit_rec.predict(h = horizon, level = levels, X_df = forecast_df)

# forecast with best model locally
get_best_model_name(cv_res_rec_local, 'rmse', by_id = True)
preds_best_df_rec_local = get_best_model_forecast(preds_df_rec, cv_res_rec_local, 'rmse', by_id = True)

# forecast with best model globally
get_best_model_name(cv_res_rec_global, 'rmse')
preds_best_df_rec_global = get_best_model_forecast(preds_df_rec, cv_res_rec_global, 'rmse')

# ** Combining forecasts --------------------------------------------------

# ** Local ----------------------------------------------------------------

preds_df_local = pl.concat([preds_best_df_no_rec_local, preds_best_df_rec_local])
preds_df_local \
    .group_by('unique_id') \
    .tk.plot_timeseries('ds', 'fcst', color_column = 'model', facet_ncol = 2, smooth = False)

# ** Global ---------------------------------------------------------------

preds_df_global = preds_best_df_no_rec_global \
    .join(preds_best_df_rec_global, on = ['unique_id', 'ds'])
plot_series(data_model_df, preds_df_global, max_insample_length = horizon * 2, engine = 'plotly').show()
