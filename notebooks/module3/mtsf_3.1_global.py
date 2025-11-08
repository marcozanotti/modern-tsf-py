# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 11: Panel Time Series Forecasting -------------------------------
# Marco Zanotti

# Goals:
# - Nested Forecasting
# - Nested Forecasting with many models
# - Global Modelling
# - Global Modelling with many models



# Packages ----------------------------------------------------------------

import re
import sys
sys.path.insert(0, 'src/Python/utils')
from utils import (
    load_data, calibrate_evaluate_plot, print_accuracy_table,
    get_best_model_name, get_best_model_forecast, select_columns
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

from neuralforecast import NeuralForecast
from neuralforecast.models import (MLP, GRU, TCN, NBEATSx, NHITS, TFT)
from neuralforecast.losses.pytorch import MQLoss



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



# NESTED FORECASTING ------------------------------------------------------

# Nested forecasting refers to the classical local approach to time series 
# forecasting, where a separate model is fitted for each individual time 
# series in a panel dataset. In this framework, the dataset typically 
# includes an identifier (e.g., unique_id), a timestamp (ds), and a target 
# variable (y). The forecasting process is nested because for each unique_id, 
# the corresponding subset of the data is extracted and a model is trained 
# specifically on that series. Forecasts are then generated independently 
# for each model.

# This approach ensures that each model is finely tuned to the unique 
# dynamics of its own series—capturing specific trends, seasonalities, or 
# structural breaks—but it can be computationally expensive and difficult 
# to scale when dealing with large collections of time series. However, 
# since each model operates independently, the process is embarrassingly 
# parallel, meaning that model training and forecasting can be easily 
# distributed across multiple cores, machines, or compute nodes without 
# the need for inter-process communication.

# While nested forecasting offers flexibility and interpretability at the 
# individual series level, it does not exploit potential relationships or 
# shared patterns across time series. These limitations have motivated the 
# development of global forecasting models, which learn from many series 
# simultaneously and provide a more scalable alternative for large-scale 
# forecasting tasks.



# GLOBAL MODELLING --------------------------------------------------------


# * Machine Leanring Models -----------------------------------------------

# ** Engines --------------------------------------------------------------

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

mlf = MLForecast(
    models = models_ml, 
    freq = '1h', 
    num_threads = -1,
    lags = [horizon],
    lag_transforms = {
        horizon: [
            ExpandingMean(),
            RollingMean(window_size = 24),
            RollingMean(window_size = 48),
            RollingMean(window_size = 96),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

mlf.preprocess(data_model_df, static_features = [], dropna = False)

# ** Evaluation -----------------------------------------------------------

cv_res_mlf = calibrate_evaluate_plot(
    mlf, df = data_model_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3, by_id = True  
)

cv_res_mlf['cv_results']

# Local accuracy
cv_res_mlf_local = cv_res_mlf['accuracy_table']
cv_res_mlf_local

# Gloabl accuracy
cv_res_mlf_global = cv_res_mlf['accuracy_table'] \
        .group_by('metric') \
        .agg(pl.col(pl.NUMERIC_DTYPES).mean())
print_accuracy_table(cv_res_mlf_global)

cv_res_mlf['plot'].show()

# ** Refitting & Forecasting ----------------------------------------------

fit_mlf = mlf.fit(df = data_model_df, prediction_intervals = intervals, static_features = [])
preds_df = fit_mlf.predict(h = horizon, level = levels, X_df = forecast_df)

# forecast with best model locally
get_best_model_name(cv_res_mlf_local, 'rmse', by_id = True)
preds_best_df_local = get_best_model_forecast(preds_df, cv_res_mlf_local, 'rmse', by_id = True)
plot_series(
    data_model_df, preds_best_df_local.drop('model'), level = levels,
    max_insample_length = horizon * 2, engine = 'plotly', 
).show()

# forecast with best model globally
get_best_model_name(cv_res_mlf_global, 'rmse')
preds_best_df_global = get_best_model_forecast(preds_df, cv_res_mlf_global, 'rmse')
plot_series(
    data_model_df, preds_best_df_global, level = levels,
    max_insample_length = horizon * 2, engine = 'plotly'
).show()


# * Deep Leanring Models --------------------------------------------------

data_model_df = select_columns(data_model_df)
forecast_df = forecast_df.select('unique_id', 'ds')

# ** Engines --------------------------------------------------------------

models_nf = [
    MLP(
        h = horizon,
        input_size = 14,
        num_layers = 2,
        hidden_size = 128,
        max_steps = 50,
        loss = MQLoss(level = levels),
        random_seed = 0
    ),
    GRU(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        encoder_n_layers = 2,
        encoder_hidden_size = 128,
        encoder_activation = 'relu',
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 50,
        loss = MQLoss(level = levels),
        random_seed = 0 
    ),
    TCN(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        kernel_size = 2, 
        dilations = [1, 7, 14],
        encoder_hidden_size = 128,
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 50,
        loss = MQLoss(level = levels),
        random_seed = 0                
    ),
    NBEATSx(
        h = horizon, 
        input_size = 30,
        stack_types = ['identity', 'trend', 'seasonality'],
        loss = MQLoss(level = levels),
        max_steps = 50,
        random_seed = 0
    ), 
    NHITS(
        h = horizon, 
        input_size = 30,
        n_freq_downsample = [2, 1, 1],
        loss = MQLoss(level = levels),
        max_steps = 50,
        random_seed = 0
    ),
    TFT(
        h = horizon,
        input_size = 30,
        hidden_size = 128,
        n_head = 2,
        loss = MQLoss(level = levels),
        max_steps = 50,
        random_seed = 0
    )
]

nf = NeuralForecast(models = models_nf, freq = '1d')

# ** Evaluation -----------------------------------------------------------

cv_res_nf = calibrate_evaluate_plot(
    nf, df = data_model_df, h = horizon, loss = 'MQLoss', 
    engine = 'plotly', max_insample_length = horizon * 2, by_id = True
)

cv_res_nf['cv_results']

# Local accuracy
cv_res_nf_local = cv_res_nf['accuracy_table']
cv_res_nf_local

# Gloabl accuracy
cv_res_nf_global = cv_res_nf['accuracy_table'] \
        .group_by('metric') \
        .agg(pl.col(pl.NUMERIC_DTYPES).mean())
print_accuracy_table(cv_res_nf_global)

cv_res_nf['plot'].show()


# ** Refitting & Forecasting ----------------------------------------------

nf.fit(df = data_model_df)

forecast_df_ns = nf.make_future_dataframe(h = horizon) # FIXME: there is a problem with future dates
preds_df = nf.predict(futr_df = forecast_df_ns).rename(lambda x: re.sub('-median', '', x))
preds_df = preds_df.with_columns(forecast_df['ds'].alias('ds'))

# forecast with best model locally
get_best_model_name(cv_res_nf_local, 'rmse', by_id = True)
preds_best_df_local = get_best_model_forecast(preds_df, cv_res_nf_local, 'rmse', by_id = True)
plot_series(
    data_model_df, preds_best_df_local.drop('model'), level = levels,
    max_insample_length = horizon * 2, engine = 'plotly', 
).show()

# forecast with best model globally
get_best_model_name(cv_res_nf_global, 'rmse')
preds_best_df_global = get_best_model_forecast(preds_df, cv_res_nf_global, 'rmse')
plot_series(
    data_model_df, preds_best_df_global, level = levels,
    max_insample_length = horizon * 2, engine = 'plotly'
).show()
