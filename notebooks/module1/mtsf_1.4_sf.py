# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture 1.3: Time Series Algorithms ---------------------------------------

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

import pickle
import sys
sys.path.insert(0, 'src/Python/utils')
from utils import (
    plot_cross_validation_plan, select_columns, calibrate_evaluate_plot,
    print_accuracy_table, get_best_model_forecast, 
    back_transform_data, back_transform_forecasts
)
import pytimetk as tk

from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse



# Data & Artifacts --------------------------------------------------------

with open('data/email/artifacts/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
feature_sets = data_loaded['feature_sets']
params = data_loaded['transform_params']


# * Recipes ---------------------------------------------------------------

# classical time series models just need the target variable and date
# so we select only those columns and the promo variable as external regressor
y_df = select_columns(data_prep_df)
y_xregs_df = select_columns(data_prep_df, '(promo)')

forecast_y_df = select_columns(forecast_df).drop('y')
forecast_xregs_df = select_columns(forecast_df, '(promo)').drop('y')

y_df.tk.plot_timeseries('ds', 'y', smooth = False)


# * Forecast Horizon ------------------------------------------------------

horizon = 7 * 8 # 8 weeks


# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
intervals = ConformalIntervals(h = horizon, n_windows = 2)
# P.S. n_windows*h should be less than the count of data elements in your time series sequence.
# P.S. Also value of n_windows should be atleast 2 or more.


# * Cross-validation Plan -------------------------------------------------

# with Nixtla's workflow there is no need to split data before 
# since validation is performed directly through the cross_validation method
# however it is always useful to visualize the validation plan 
# for that one can just cross-validate a naive model to obtain the 
# validation plan (cutoffs dates)
plot_cross_validation_plan(y_df, freq = '1d', h = horizon, n_windows = 1, step_size = 1)

plot_cross_validation_plan(y_df, freq = '1d', h = horizon, n_windows = 6, step_size = 14)



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
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_baseline = calibrate_evaluate_plot(
    sf_baseline, y_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2, plot_level = True
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
    df = y_df, h = horizon,
    prediction_intervals = intervals, 
    level = levels
)
fcst_df_baseline

sf_baseline.plot(
    y_df, fcst_df_baseline,
    max_insample_length = horizon * 2,
    engine = 'plotly'
).show()

for nm in sf_baseline.models:
    sf_baseline.plot(
        y_df, 
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
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

# ARIMA models without external regressors
cv_res_arima = calibrate_evaluate_plot(
    object = sf_arima, df = y_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_arima['cv_results']
cv_res_arima['accuracy_table']
cv_res_arima['plot'].show()

# ARIMA models with external regressors
cv_res_arima_xregs = calibrate_evaluate_plot(
    object = sf_arima, df = y_xregs_df, 
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
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_ets = calibrate_evaluate_plot(
    object = sf_ets, df = y_df, 
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
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_theta = calibrate_evaluate_plot(
    object = sf_theta, df = data_prep_df, 
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
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_ms = calibrate_evaluate_plot(
    object = sf_ms, df = y_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_ms['cv_results']
cv_res_ms['accuracy_table']
cv_res_ms['plot'].show()



# Prophet -----------------------------------------------------------------

from prophet import Prophet

# manual train test split
# Prophet needs a pandas DataFrame
prophet_y_df = y_df.to_pandas()
train_prophet_df = prophet_y_df.head(-horizon)
test_prophet_df = prophet_y_df.tail(horizon)

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
        preds_prophet[['ds', 'yhat']].rename(columns = {'yhat': 'Prophet'}),
        on = 'ds'
    ),
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)

# plot
StatsForecast.plot(
    prophet_y_df, 
    preds_prophet[['ds', 'yhat']] \
        .rename(columns = {'yhat': 'Prophet'}) \
        .assign(unique_id = 'email_subscribers'), 
    engine = 'plotly'
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
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_mfles = calibrate_evaluate_plot(
    object = sf_mfles, df = y_df, 
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
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

# with external regressors
cv_res_ts = calibrate_evaluate_plot(
    object = sf_ts, df = y_xregs_df,
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_ts['cv_results']
print_accuracy_table(cv_res_ts['accuracy_table'], 'min')
cv_res_ts['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

fcst_df_ts = sf_ts.forecast(
    df = y_xregs_df, X_df = forecast_xregs_df, h = horizon, 
    prediction_intervals = intervals, level = levels
)
fcst_df_ts

sf_ts.plot(
    y_xregs_df, fcst_df_ts,
    max_insample_length = horizon * 2,
    engine = 'plotly'
).show()

# * Select Best Model -----------------------------------------------------

fcst_best_df = get_best_model_forecast(fcst_df_ts, cv_res_ts['accuracy_table'], 'rmse')
sf_ts.plot(
    y_xregs_df, fcst_best_df,
    max_insample_length = horizon * 2,
    engine = 'plotly', 
    level = levels
).show()

# * Back-transform --------------------------------------------------------

back_df = back_transform_data(y_xregs_df, params)
back_fcst_df = back_transform_forecasts(fcst_best_df, params)
sf_ts.plot(
    back_df, back_fcst_df, 
    max_insample_length = horizon * 2,
    engine = 'plotly', 
    level = levels
).show()