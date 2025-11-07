# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture 2.5: Recursive Time Series Algorithms ----------------------------

# Goals:
# - Recursivity



# Packages ----------------------------------------------------------------

import sys
sys.path.insert(0, 'src/Python/utils')
from utils import (
    load_data, log_interval, standardize, 
    calibrate_evaluate_plot, plot_cross_validation_plan,
    print_accuracy_table, get_best_model_forecast, get_best_model_name,
    back_transform_data, back_transform_forecasts
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
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from cubist import Cubist
from sklearn.neural_network import MLPRegressor



# Data --------------------------------------------------------------------

email_df = load_data('data/email/', 'email_prep', ext = '.parquet')
email_df.glimpse()


# * Data Preparation ------------------------------------------------------

# Pre-processing Data

email_df.tk.plot_timeseries('ds', 'y', smooth = False)

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
    .select('unique_id', 'ds', 'y')

params = {
    'lower_bound': y_lb,
    'upper_bound': y_ub,
    'offset': y_offset,
    'mean_y': y_mean,
    'stdev_y': y_std
}

data_prep_df.tk.plot_timeseries('ds', 'y', smooth = False)


# * Forecast Horizon ------------------------------------------------------

horizon = 7 * 8 # 8 weeks


# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
intervals = PredictionIntervals(h = horizon, n_windows = 2)


# * Feature Engineering ---------------------------------------------------

# We use Nixtla's direct feature engineering utilities to create 
# time based features (calendar, trend, and fourier).
# However, we do not manually create any lag or rolling feature, but 
# we make use of MLForecast capabilities. In this way, the lags will be
# internally created and recursivity will be automatically applied.

# custom feature function
def is_weekend(times):
    dow = times.dt.weekday()
    return dow >= 6

features = [
    trend,
    partial(fourier, season_length = 7, k = 1),
    partial(fourier, season_length = 14, k = 1),
    partial(fourier, season_length = 30, k = 1),
    partial(time_features, features = ['day', 'weekday', 'week', 'month', 'quarter', is_weekend]),
]

# also separate into modelling & forecast datasets
data_model_df, forecast_df = pipeline(
    data_prep_df, features = features, freq = '1d', h = horizon,
)

data_model_df.glimpse()
forecast_df.glimpse()


# * Cross-validation Plan -------------------------------------------------

plot_cross_validation_plan(
    data_model_df, freq = '1d', h = horizon, n_windows = 1, step_size = 1
)



# Machine Learning Models -------------------------------------------------


# * Non-Recursive ---------------------------------------------------------

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
    GradientBoostingRegressor(
        loss = 'squared_error',
        n_estimators = 100, 
        learning_rate = 0.1,
        random_state = 0
    ),
    AdaBoostRegressor(
        loss = 'square',
        n_estimators = 100,
        learning_rate = 1.0,
        random_state = 0
    ), 
    XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'reg:squarederror',
        random_state = 0
    ),
    Cubist(
        n_rules = 100, 
        n_committees = 10,
        neighbors = 7,
        random_state = 0
    ),
    MLPRegressor(
        hidden_layer_sizes = (10,),
        learning_rate_init = 0.1,
        alpha = 0.1,
        activation = 'relu',
        solver = 'adam',
        random_state = 0
    )
]

mlf_no_rec = MLForecast(
    models = models_ml, 
    freq = '1d', 
    num_threads = -1,
    lags = [horizon],
    lag_transforms = {
        horizon: [
            ExpandingMean(),
            RollingMean(window_size = 7),
            RollingMean(window_size = 14),
            RollingMean(window_size = 30),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

mlf_no_rec.preprocess(data_model_df, static_features = [], dropna = False)


# * Recursive -------------------------------------------------------------

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
    'GradientBoostingRegressor_rec': GradientBoostingRegressor(
        loss = 'squared_error',
        n_estimators = 100, 
        learning_rate = 0.1,
        random_state = 0
    ),
    'AdaBoostRegressor_rec': AdaBoostRegressor(
        loss = 'square',
        n_estimators = 100,
        learning_rate = 1.0,
        random_state = 0
    ), 
    'XGBRegressor_rec': XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'reg:squarederror',
        random_state = 0
    ),
    'Cubist_rec': Cubist(
        n_rules = 100, 
        n_committees = 10,
        neighbors = 7,
        random_state = 0
    ),
    'MLP_rec': MLPRegressor(
        hidden_layer_sizes = (10,),
        learning_rate_init = 0.1,
        alpha = 0.1,
        activation = 'relu',
        solver = 'adam',
        random_state = 0
    )
}

mlf_rec = MLForecast(
    models = models_ml_rec, 
    freq = '1d', 
    num_threads = -1,
    lags = [1, 2, 7, 14, 30],
    lag_transforms = {
        1: [
            ExpandingMean(), 
            ExponentiallyWeightedMean(alpha = 0.3),
            RollingMean(window_size = 7)
        ],
        7: [
            RollingMean(window_size = 7, min_samples = 1)
        ],
        30: [
            RollingMean(window_size = 14, min_samples = 1)
        ]
    }
)

mlf_rec.preprocess(data_model_df, static_features = [], dropna = False)


# * Evaluation ------------------------------------------------------------

cv_res_no_rec = calibrate_evaluate_plot(
    mlf_no_rec, df = data_model_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_no_rec['cv_results']
print_accuracy_table(cv_res_no_rec['accuracy_table'])
cv_res_no_rec['plot'].show()

cv_res_rec = calibrate_evaluate_plot(
    mlf_rec, df = data_model_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_rec['cv_results']
print_accuracy_table(cv_res_rec['accuracy_table'])
cv_res_rec['plot'].show()

print_accuracy_table(cv_res_no_rec['accuracy_table'])
print_accuracy_table(cv_res_rec['accuracy_table'])

cv_res_accuracy = cv_res_no_rec['accuracy_table'] \
    .join(cv_res_rec['accuracy_table'], on = 'metric')
cols = ['metric'] + sorted([c for c in cv_res_accuracy.columns if c != 'metric'])
cv_res_accuracy = cv_res_accuracy.select(cols)
print_accuracy_table(cv_res_accuracy)


# * Refitting & Forecasting -----------------------------------------------

fit_no_rec = mlf_no_rec.fit(df = data_model_df, prediction_intervals = intervals, static_features = [])
preds_df_no_rec = fit_no_rec.predict(h = horizon, level = levels, X_df = forecast_df)

fit_rec = mlf_rec.fit(df = data_model_df, prediction_intervals = intervals, static_features = [])
preds_df_rec = fit_rec.predict(h = horizon, level = levels, X_df = forecast_df)

preds_df = preds_df_no_rec.join(preds_df_rec, on = ['unique_id', 'ds'])

plot_series(data_model_df, preds_df, max_insample_length = horizon * 2, engine = 'plotly').show()


# * Select Best Model -----------------------------------------------------

cv_res_accuracy = cv_res_no_rec['accuracy_table'] \
    .join(cv_res_rec['accuracy_table'], on = 'metric')
cols = ['metric'] + sorted([c for c in cv_res_accuracy.columns if c != 'metric'])
cv_res_accuracy = cv_res_accuracy.select(cols)

get_best_model_name(cv_res_accuracy, metric = 'mae')
get_best_model_name(cv_res_accuracy, metric = 'rmse')
get_best_model_name(cv_res_accuracy, metric = 'mape')
print_accuracy_table(cv_res_accuracy)

preds_best_df = get_best_model_forecast(preds_df, cv_res_accuracy, 'rmse')
preds_best_df = preds_best_df.select(preds_best_df.columns[:7])
plot_series(
    data_model_df, preds_best_df, level = levels,
    max_insample_length = horizon * 2, engine = 'plotly'
).show()


# * Back-transform --------------------------------------------------------

back_df = back_transform_data(data_model_df, params)
back_fcst_best_df = back_transform_forecasts(preds_best_df, params)
plot_series(
    back_df, back_fcst_best_df, level = levels,
    max_insample_length = horizon * 3, engine = 'plotly'
).show()
