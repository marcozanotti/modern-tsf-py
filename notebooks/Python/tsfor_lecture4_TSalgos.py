# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 4: Time Series Algorithms ---------------------------------------
# Marco Zanotti

# Goals:
# - Baseline Models
# - SARIMAX
# - ETS
# - Theta
# - TBATS
# - MSTL
# - PROPHET
# - BONUS: Intermittent Demand Models



# Packages ----------------------------------------------------------------

import os
import pickle

import numpy as np
import pandas as pd
import random
# import polars as pl
# import pytimetk as tk

from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse

import python_extensions as pex

pd.set_option("display.max_rows", 3)
os.environ['NIXTLA_ID_AS_COL'] = '1'



# Data & Artifacts --------------------------------------------------------

with open('artifacts/Python/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
feature_sets = data_loaded['feature_sets']
params = data_loaded['transform_params']

StatsForecast.plot(data_prep_df).show()

# * Forecast Horizon ------------------------------------------------------
horizon = 7 * 8 # 8 weeks

# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
# Conformal intervals
intervals = ConformalIntervals(h = horizon, n_windows = 2)
# P.S. n_windows*h should be less than the count of data elements in your time series sequence.
# P.S. Also value of n_windows should be atleast 2 or more.

# * Cross-validation Plan -------------------------------------------------

# with Nixtla's workflow there is no need to split data before 
# since validation is performed directly through the cross_validation method
# however it is always useful to visualize the validation plan 
# for that one can just cross-validate a naive model to obtain the 
# validation plan (cutoffs dates)
pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 6, step_size = 14, 
    engine = 'matplotlib'
)

# * External Regressors ---------------------------------------------------

y_df = pex.select_columns(data_prep_df)
y_xregs_df = pex.select_columns(data_prep_df, '(event)')



# S-NAIVE & AVERAGES ------------------------------------------------------

# Naive
# Seasonal Naive
# Averages
from statsforecast.models import (
    HistoricAverage,
    Naive, 
    SeasonalNaive,
    RandomWalkWithDrift,
    WindowAverage,
    SeasonalWindowAverage
)

# * Engines ---------------------------------------------------------------

models_baseline = [
    HistoricAverage(),
    Naive(),
    SeasonalNaive(season_length = 7),
    RandomWalkWithDrift(),
    WindowAverage(window_size = 7), 
    SeasonalWindowAverage(season_length = 7, window_size = 8)
]
# Instantiate StatsForecast class
sf_baseline = StatsForecast(
    models = models_baseline,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_baseline = pex.calibrate_evaluate_plot(
    sf_baseline, data_prep_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)

cv_res_baseline['cv_results']
cv_res_baseline['accuracy_table']
cv_res_baseline['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

# if you need model parameters use .fit and .predict 
# otherwise use .forecast (optimized to run also on clusters)
# P.S. specify fitted = True to store the fitted values

# fit_baseline = sf_baseline.fit(
#     df = data_prep_df, prediction_intervals = intervals
# )
# preds_df_baseline = fit_baseline.predict(
#     h = horizon, level = levels
# )
fcst_df_baseline = sf_baseline.forecast(
    df = data_prep_df, h = horizon,
    prediction_intervals = intervals, 
    level = levels
)
fcst_df_baseline

sf_baseline.plot(
    data_prep_df, fcst_df_baseline,
    max_insample_length = horizon * 2,
    engine = 'plotly'
).show()

for nm in sf_baseline.models:
    sf_baseline.plot(
        data_prep_df, 
        fcst_df_baseline,
        models = [str(nm)], 
        level = levels,
        max_insample_length = horizon * 2,
        engine = 'plotly'
    ).show()



# S-ARIMA-X ---------------------------------------------------------------

# Seasonal Regression with ARIMA Errors and External Regressors
# yt = alpha * L(yt)^k +  beta L(yt)^s + et + gamma * L(et)^k + delta * L(xt)^k

# ARIMA is a simple algorithm that relies on Linear Regression
# Strengths:
# - Automated Differencing
# - Automated Parameter Search (auto_arima)
# - Single seasonality modeling included
# - Recursive Lag Forecasting
# Weaknesses:
# - Only single seasonality by default (XREGs can help go beyond single seasonality)
# - Becomes erratic with too many lags
# - Requires Expensive Parameter Search

from statsforecast.models import (
    ARIMA,
    AutoRegressive, 
    AutoARIMA
)

# * Engines ---------------------------------------------------------------

models_arima = [
    ARIMA(order = (1, 1, 1), season_length = 7, seasonal_order = (1, 1, 1)),
    AutoRegressive(lags = [1, 7, 14, 30]),
    AutoARIMA(season_length = 7)
]
sf_arima = StatsForecast(
    models = models_arima,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

# ARIMA models without external regressors
cv_res_arima = pex.calibrate_evaluate_plot(
    class_object = sf_arima, data = y_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_arima['cv_results']
cv_res_arima['accuracy_table']
cv_res_arima['plot'].show()

# ARIMA models with external regressors
cv_res_arima_xregs = pex.calibrate_evaluate_plot(
    class_object = sf_arima, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_arima_xregs['cv_results']
cv_res_arima_xregs['accuracy_table']
cv_res_arima_xregs['plot'].show()



# EXPONENTIAL SMOOTHING (ETS) ---------------------------------------------

# Error, Trend & Seasonality (Holt-Winters Seasonal)

# - Automatic forecasting method based on Exponential Smoothing
# - Single Seasonality
# - Cannot use XREGs (purely univariate)

from statsforecast.models import (
    SimpleExponentialSmoothing,
    SimpleExponentialSmoothingOptimized,
    SeasonalExponentialSmoothing,
    SeasonalExponentialSmoothingOptimized,
    Holt,
    HoltWinters,
    AutoETS,
    AutoCES
)

# * Engines ---------------------------------------------------------------

models_ets = [
    SimpleExponentialSmoothing(alpha = 0.7),
    SimpleExponentialSmoothingOptimized(),
    SeasonalExponentialSmoothing(season_length = 7, alpha = 0.7),
    SeasonalExponentialSmoothingOptimized(season_length = 7),
    Holt(season_length = 7, error_type = 'A'),
    HoltWinters(season_length = 7, error_type = 'A'),
    AutoETS(season_length = 7, model = 'ZZZ', damped = True),
    AutoCES(season_length = 7)
]
sf_ets = StatsForecast(
    models = models_ets,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_ets = pex.calibrate_evaluate_plot(
    class_object = sf_ets, data = data_prep_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_ets['cv_results']
cv_res_ets['accuracy_table']
cv_res_ets['plot'].show()



# Theta -------------------------------------------------------------------

from statsforecast.models import (
    Theta,
    OptimizedTheta,
    DynamicTheta,
    DynamicOptimizedTheta
)

# * Engines ---------------------------------------------------------------

models_theta = [
    Theta(season_length = 7, decomposition_type = 'multiplicative'),
    OptimizedTheta(season_length = 7),
    DynamicTheta(season_length = 7, decomposition_type = 'multiplicative'),
    DynamicOptimizedTheta(season_length = 7)
]
sf_theta = StatsForecast(
    models = models_theta,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_theta = pex.calibrate_evaluate_plot(
    class_object = sf_theta, data = data_prep_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_theta['cv_results']
cv_res_theta['accuracy_table']
cv_res_theta['plot'].show()



# Multiple Seasonality ----------------------------------------------------

# MSTL
# Seasonal & Trend Decomposition using LOESS Models

# - Uses seasonal decomposition to model trend & seasonality separately
#   - Trend modeled with ARIMA or ETS
#   - Seasonality modeled with Seasonal Naive (SNAIVE)
# - Can handle multiple seasonality
# - ARIMA version accepts XREGS, ETS does not

# TBATS
# Exponential Smoothing with Box-Cox transformation, ARMA errors, Trend and Seasonality

# - Multiple Seasonality Model
# - Extension of ETS for complex seasonality
# - Automatic
# - Does not support XREGS
# - Computationally low (often)

from statsforecast.models import (
    MSTL, 
    TBATS, 
    AutoTBATS,
    AutoETS, AutoARIMA
)

# * Engines ---------------------------------------------------------------

models_ms = [
    MSTL(season_length = [7, 30], trend_forecaster = AutoETS(model = 'ZZN', damped = True)),
    MSTL(season_length = [7, 30], trend_forecaster = AutoARIMA(), alias = 'MSTL_ARIMA'),
    TBATS(season_length = [7, 30, 365], use_boxcox = False, use_damped_trend = True),
    AutoTBATS(season_length = [7, 30, 365])
]
sf_ms = StatsForecast(
    models = models_ms,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

# try with and without xregs
y_df
y_xregs_df

cv_res_ms = pex.calibrate_evaluate_plot(
    class_object = sf_ms, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_ms['cv_results']
cv_res_ms['accuracy_table']
cv_res_ms['plot'].show()



# Prophet -----------------------------------------------------------------

from prophet import Prophet

# manual train test split
train_prophet_df = y_df.head(-horizon)
test_prophet_df = y_df.tail(horizon)

# * Engines ---------------------------------------------------------------

help(Prophet)
model_prophet = Prophet(
    growth = 'linear', 
    n_changepoints = 10, 
    changepoint_range = 0.9,
    yearly_seasonality = True, 
    weekly_seasonality = True
)

# * Evaluation ------------------------------------------------------------

# fit
model_prophet.fit(train_prophet_df)

# forecast dates
forecast_prohet_df = model_prophet.make_future_dataframe(
    periods = horizon, freq = 'D'
)

# predict
preds_prophet = model_prophet.predict(forecast_prohet_df)
preds_prophet
preds_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# evaluate
evaluate(
    test_prophet_df.merge(
        preds_prophet[['ds', 'yhat']] \
            .rename(columns = {'yhat': 'Prophet'}),
        on = 'ds'
    ),
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)

# plot
StatsForecast.plot(
    y_df, 
    preds_prophet[['ds', 'yhat']] \
        .rename(columns = {'yhat': 'Prophet'}) \
        .assign(unique_id = 'subscribers')
).show()

model_prophet.plot(preds_prophet, uncertainty = True)
model_prophet.plot_components(preds_prophet)



# MFLES -------------------------------------------------------------------

# A method to forecast time series based on Gradient Boosted Time 
# Series Decomposition which treats traditional decomposition as 
# the base estimator in the boosting process. Unlike normal gradient 
# boosting, slight learning rates are applied at the component 
# level (trend/seasonality/exogenous).

# The method derives its name from some of the underlying estimators 
# that can enter into the boosting procedure, specifically: a simple 
# Median, Fourier functions for seasonality, a simple/piecewise Linear 
# trend, and Exponential Smoothing.
from statsforecast.models import (
    MFLES,
    AutoMFLES
)

# * Engines ---------------------------------------------------------------

models_mfles = [
    MFLES(season_length = 7),
    AutoMFLES(test_size = 28, n_windows = 2, season_length = 7, metric = 'mse')
]
sf_mfles = StatsForecast(
    models = models_mfles,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_mfles = pex.calibrate_evaluate_plot(
    class_object = sf_mfles, data = data_prep_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_mfles['cv_results']
cv_res_mfles['accuracy_table']
cv_res_mfles['plot'].show()



# TS Models' Performance Comparison ---------------------------------------

from statsforecast.models import (
    WindowAverage, 
    SeasonalWindowAverage,
    AutoARIMA,
    AutoETS,
    AutoCES,
    DynamicOptimizedTheta,
    MSTL,
    AutoTBATS
)

# * Engines ---------------------------------------------------------------

models_ts = [
    WindowAverage(window_size = 7), 
    SeasonalWindowAverage(season_length = 7, window_size = 8),
    AutoARIMA(season_length = 7),
    AutoETS(season_length = 7),
    AutoCES(season_length = 7),
    DynamicOptimizedTheta(season_length = 7),
    MSTL(season_length = [7, 30], trend_forecaster = AutoETS(model = 'ZZN', damped = True)),
    MSTL(season_length = [7, 30], trend_forecaster = AutoARIMA(), alias = 'MSTL_ARIMA'),
    AutoTBATS(season_length = [7, 30, 365])
]
sf_ts = StatsForecast(
    models = models_ts,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

# with external regressors
cv_res_ts = pex.calibrate_evaluate_plot(
    object = sf_ts, data = y_xregs_df,
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_ts['cv_results']
cv_res_ts['accuracy_table'].print_accuracy_table('min')
cv_res_ts['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

fcst_df_ts = sf_ts.forecast(
    df = y_xregs_df, 
    X_df = forecast_df.reindex(columns = ['unique_id', 'ds', 'event']),
    h = horizon, 
    prediction_intervals = intervals, 
    level = levels
)
fcst_df_ts

sf_ts.plot(
    data_prep_df, fcst_df_ts,
    max_insample_length = horizon * 2,
    engine = 'plotly'
).show()

# * Select Best Model -----------------------------------------------------

fcst_best_df = fcst_df_ts \
    .get_best_model_forecast(
        cv_res_ts['accuracy_table'], 
        'rmse'
    )
sf_ts.plot(
    data_prep_df, fcst_best_df,
    max_insample_length = horizon * 2,
    engine = 'plotly', 
    level = levels
).show()

# * Back-transform --------------------------------------------------------

data_back_dict = pex.back_transform_data(data_prep_df, params, fcst_best_df)
sf_ts.plot(
    data_back_dict['data_back'], 
    data_back_dict['forecasts_back'], 
    max_insample_length = horizon * 2,
    engine = 'plotly', 
    level = levels
).show()



# BONUS -------------------------------------------------------------------
# Sparse & Intermittent ---------------------------------------------------

from statsforecast.models import (
    CrostonClassic as Croston, 
    CrostonOptimized,
    CrostonSBA,
    TSB,
    ADIDA, 
    IMAPA
)

# * Engines ---------------------------------------------------------------

models_inter = [
    Croston(),
    CrostonOptimized(),
    CrostonSBA(),
    TSB(alpha_d = 0.5, alpha_p = 0.5),
    ADIDA(),
    IMAPA()
]
sf_inter = StatsForecast(
    models = models_inter,
    freq = 'D', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

# create a intermittent dataframe
subscribers_df = pd.read_csv(
    "data/subscribers/subscribers.csv", 
    parse_dates = ['optin_time']
) \
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
    .rename(columns = {'observed_clean': 'value'}) \
    .rename(
        columns = {
            'id': 'unique_id',
            'date': 'ds',
            'value': 'y'
        }
    ) \
        .reset_index() \
        .drop('index', axis = 1)
subscribers_df

random.seed(1992)
inter_df = pex.to_intermittent(subscribers_df, 0.90)
StatsForecast.plot(inter_df).show()

cv_res_inter = pex.calibrate_evaluate_plot(
    class_object = sf_inter, data = inter_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_inter['cv_results']
cv_res_inter['accuracy_table']
cv_res_inter['plot'].show()



# # XX ------------------------------------------------------

# from statsforecast.models import (
    
# )

# # * Engines ---------------------------------------------------------------

# models_xx = [
    
# ]
# sf_xx = StatsForecast(
#     models = models_xx,
#     freq = 'D', 
#     n_jobs = -1,
# )

# # * Evaluation ------------------------------------------------------------

# cv_res_xx = pex.calibrate_evaluate_plot(
#     class_object = sf_xx, data = data_prep_df, 
#     h = horizon, prediction_intervals = intervals, level = levels,
#     engine = 'plotly', max_insample_length = horizon * 2  
# )
# cv_res_xx['cv_results']
# cv_res_xx['accuracy_table']
# cv_res_xx['plot'].show()