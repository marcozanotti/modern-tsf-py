# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture 1.5: Machine Learning Algorithms ----------------------------------

# Goals:
# - Linear Regression
# - Elastic Net
# - MARS
# - SVM
# - KNN
# - Random Forest
# - XGBoost, Light GBM, CAT Boost
# - Cubist
# - Neural Networks

# Challenges:
# - Challenge 2 - Testing New Forecasting Algorithms



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

from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from utilsforecast.plotting import plot_series



# Data & Artifacts --------------------------------------------------------

with open('data/email/artifacts/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
feature_sets = data_loaded['feature_sets']
params = data_loaded['transform_params']


# * Recipes ---------------------------------------------------------------

# in ML modeling we can test various feature sets
y_df = select_columns(data_prep_df)
y_base_df = data_prep_df.select(feature_sets['base'])
y_wave_df = data_prep_df.select(feature_sets['wave'])
y_lag_df = data_prep_df.select(feature_sets['lag']).drop_nulls()

forecast_y_df = select_columns(forecast_df).drop('y')
forecast_y_base_df = forecast_df.select(feature_sets['base']).drop('y')
forecast_y_wave_df = forecast_df.select(feature_sets['wave']).drop('y')
forecast_y_lag_df = forecast_df.select(feature_sets['lag']).drop('y')

y_df.tk.plot_timeseries('ds', 'y', smooth = False)


# * Forecast Horizon ------------------------------------------------------
horizon = 7 * 8 # 8 weeks


# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
intervals = PredictionIntervals(h = horizon, n_windows = 2)
# P.S. n_windows*h should be less than the count of data elements in your time series sequence.
# P.S. Also value of n_windows should be atleast 2 or more.


# * Cross-validation Plan -------------------------------------------------

plot_cross_validation_plan(y_df, freq = '1d', h = horizon, n_windows = 1, step_size = 1)



# Linear Regression -------------------------------------------------------

# - Baseline model for ML
from sklearn.linear_model import LinearRegression


# * Engines ---------------------------------------------------------------

models_lr = [LinearRegression()]
mlf_lr = MLForecast(models = models_lr, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_lr_base = calibrate_evaluate_plot(
    mlf_lr, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_lr_base['cv_results']
cv_res_lr_base['accuracy_table']
cv_res_lr_base['plot'].show()

cv_res_lr_wave = calibrate_evaluate_plot(
    mlf_lr, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_lr_wave['cv_results']
cv_res_lr_wave['accuracy_table']
cv_res_lr_wave['plot'].show()

cv_res_lr_lag = calibrate_evaluate_plot(
    mlf_lr, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_lr_lag['cv_results']
cv_res_lr_lag['accuracy_table']
cv_res_lr_lag['plot'].show()


# * Refitting & Forecasting -----------------------------------------------

fit_lr = mlf_lr.fit(df = y_base_df, prediction_intervals = intervals, static_features = [])
fit_lr.models_['LinearRegression'].intercept_
fit_lr.models_['LinearRegression'].coef_

preds_df_lr = fit_lr.predict(h = horizon, level = levels, X_df = forecast_y_base_df)
preds_df_lr

plot_series(
    y_base_df, preds_df_lr, level = levels,
    max_insample_length = horizon * 3, engine = 'plotly'
).show()



# Elastic Net -------------------------------------------------------

# - Strengths: Very good for trend
# - Weaknesses: Not as good for complex patterns (i.e. seasonality)
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# * Engines ---------------------------------------------------------------

models_elanet = [
    Ridge(),
    Lasso(alpha = 0.01),
    ElasticNet(l1_ratio = 0.5, alpha = 0.01)    
]
mlf_elanet = MLForecast(models = models_elanet, freq = '1d', num_threads = 1)

# * Evaluation ------------------------------------------------------------

cv_res_elanet_base = calibrate_evaluate_plot(
    mlf_elanet, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_elanet_base['cv_results']
cv_res_elanet_base['accuracy_table']
cv_res_elanet_base['plot'].show()

cv_res_elanet_wave = calibrate_evaluate_plot(
    mlf_elanet, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_elanet_wave['cv_results']
cv_res_elanet_wave['accuracy_table']
cv_res_elanet_wave['plot'].show()

cv_res_elanet_lag = calibrate_evaluate_plot(
    mlf_elanet, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_elanet_lag['cv_results']
cv_res_elanet_lag['accuracy_table']
cv_res_elanet_lag['plot'].show()



# MARS --------------------------------------------------------------------

# Multiple Adaptive Regression Splines

# - Strengths: Best algorithm for modeling trend
# - Weaknesses:
#   - Not good for complex patterns (i.e. seasonality)
#   - Don't combine with splines! MARS makes splines.

# FIXME: installation error and archived repo, you need to use Python 3.6
# https://github.com/scikit-learn-contrib/py-earth
# pip install sklearn-contrib-py-earth
from pyearth import Earth


# * Engines ---------------------------------------------------------------

models_mars = [Earth()]
mlf_mars = MLForecast(models = models_mars, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_mars_base = calibrate_evaluate_plot(
    mlf_mars, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_mars_base['cv_results']
cv_res_mars_base['accuracy_table']
cv_res_mars_base['plot'].show()

cv_res_mars_wave = calibrate_evaluate_plot(
    mlf_mars, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_mars_wave['cv_results']
cv_res_mars_wave['accuracy_table']
cv_res_mars_wave['plot'].show()

cv_res_mars_lag = calibrate_evaluate_plot(
    mlf_mars, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_mars_lag['cv_results']
cv_res_mars_lag['accuracy_table']
cv_res_mars_lag['plot'].show()



# SVM ---------------------------------------------------------------------

# Support Vector Machines

# Strengths: Well-rounded algorithm
# Weaknesses: Needs tuned or can overfit and can be computationally inefficient
# - Strengths: Very good for trend
# - Weaknesses: Not as good for complex patterns (i.e. seasonality)
from sklearn.svm import SVR


# * Engines ---------------------------------------------------------------

models_svm = [
    # SVR(kernel = 'linear'), # very slow !!!!!!
    # SVR(kernel = 'poly'), # very slow !!!!!!
    SVR(kernel = 'rbf')
]
mlf_svm = MLForecast(models = models_svm, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_svm_base = calibrate_evaluate_plot(
    mlf_svm, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_svm_base['cv_results']
cv_res_svm_base['accuracy_table']
cv_res_svm_base['plot'].show()

cv_res_svm_wave = calibrate_evaluate_plot(
    mlf_svm, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_svm_wave['cv_results']
cv_res_svm_wave['accuracy_table']
cv_res_svm_wave['plot'].show()

cv_res_svm_lag = calibrate_evaluate_plot(
    mlf_svm, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_svm_lag['cv_results']
cv_res_svm_lag['accuracy_table']
cv_res_svm_lag['plot'].show()



# KNN ---------------------------------------------------------------------

# K Neighrest Neighbors

# - Strengths: Uses neighboring points to estimate
# - Weaknesses: Cannot predict beyond the maximum/minimum target (e.g. increasing trend)
# - Solution: Model trend separately (if needed).
#   - Can combine with ARIMA, Linear Regression, knn, or Prophet
from sklearn.neighbors import KNeighborsRegressor

# * Engines ---------------------------------------------------------------

models_knn = [KNeighborsRegressor(n_neighbors = 30)]
mlf_knn = MLForecast(models = models_knn, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_knn_base = calibrate_evaluate_plot(
    mlf_knn, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_knn_base['cv_results']
cv_res_knn_base['accuracy_table']
cv_res_knn_base['plot'].show()

cv_res_knn_wave = calibrate_evaluate_plot(
    mlf_knn, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_knn_wave['cv_results']
cv_res_knn_wave['accuracy_table']
cv_res_knn_wave['plot'].show()

cv_res_knn_lag = calibrate_evaluate_plot(
    mlf_knn, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_knn_lag['cv_results']
cv_res_knn_lag['accuracy_table']
cv_res_knn_lag['plot'].show()



# GAUSSIAN PROCESS REGRESSION ---------------------------------------------

from sklearn.gaussian_process import GaussianProcessRegressor

# * Engines ---------------------------------------------------------------

models_gp = [GaussianProcessRegressor()]
mlf_gp = MLForecast(models = models_gp, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_gp_base = calibrate_evaluate_plot(
    mlf_gp, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_gp_base['cv_results']
cv_res_gp_base['accuracy_table']
cv_res_gp_base['plot'].show()

cv_res_gp_wave = calibrate_evaluate_plot(
    mlf_gp, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_gp_wave['cv_results']
cv_res_gp_wave['accuracy_table']
cv_res_gp_wave['plot'].show()

cv_res_gp_lag = calibrate_evaluate_plot(
    mlf_gp, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_gp_lag['cv_results']
cv_res_gp_lag['accuracy_table']
cv_res_gp_lag['plot'].show()



# REGRESSION TREE ---------------------------------------------------------

# - Baseline Tree model
from sklearn.tree import DecisionTreeRegressor


# * Engines ---------------------------------------------------------------

models_tree = [
    DecisionTreeRegressor(
        criterion = 'squared_error', 
        splitter = 'best',
        max_depth = None,
        min_samples_split = 2
    )
]
mlf_tree = MLForecast(models = models_tree, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_tree_base = calibrate_evaluate_plot(
    mlf_tree, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_tree_base['cv_results']
cv_res_tree_base['accuracy_table']
cv_res_tree_base['plot'].show()

cv_res_tree_wave = calibrate_evaluate_plot(
    mlf_tree, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_tree_wave['cv_results']
cv_res_tree_wave['accuracy_table']
cv_res_tree_wave['plot'].show()

cv_res_tree_lag = calibrate_evaluate_plot(
    mlf_tree, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_tree_lag['cv_results']
cv_res_tree_lag['accuracy_table']
cv_res_tree_lag['plot'].show()



# BAGGING & RANDOM FOREST -------------------------------------------------

# - Strengths: Can model seasonality very well
# - Weaknesses:
#   - Cannot predict beyond the maximum/minimum target (e.g. increasing trend)
# - Solution: Model trend separately (if needed).
#   - Can combine with ARIMA, Linear Regression, Mars, or Prophet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

# * Engines ---------------------------------------------------------------

models_rf = [
    BaggingRegressor(
        n_estimators = 100,
        max_samples = 10,
        max_features = 10,
        bootstrap = True,
        random_state = 0
    ),
    RandomForestRegressor(
        n_estimators = 100,
        criterion = 'squared_error',
        max_depth = None,
        min_samples_split = 2,
        max_features = 'sqrt',
        random_state = 0
    )
]
mlf_rf = MLForecast(models = models_rf, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_rf_base = calibrate_evaluate_plot(
    mlf_rf, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_rf_base['cv_results']
cv_res_rf_base['accuracy_table']
cv_res_rf_base['plot'].show()

cv_res_rf_wave = calibrate_evaluate_plot(
    mlf_rf, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_rf_wave['cv_results']
cv_res_rf_wave['accuracy_table']
cv_res_rf_wave['plot'].show()

cv_res_rf_lag = calibrate_evaluate_plot(
    mlf_rf, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_rf_lag['cv_results']
cv_res_rf_lag['accuracy_table']
cv_res_rf_lag['plot'].show()



# BOOSTING ----------------------------------------------------------------

# Gradient Boosting
# AdaBOOST
# XGBOOST

# LIGHT GBM
# https://lightgbm.readthedocs.io/en/latest/
# https://github.com/microsoft/LightGBM

# CAT BOOST
# https://catboost.ai/en/docs/
# https://github.com/catboost/catboost

# - Strengths: Best for seasonality & complex patterns
# - Weaknesses:
#   - Cannot predict beyond the maximum/minimum target (e.g. increasing trend)
# - Solution: Model trend separately (if needed).
#   - Can combine with ARIMA, Linear Regression, Mars, or Prophet
#   - prophet_boost & arima_boost: Do this
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# * Engines ---------------------------------------------------------------

models_boost = [
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
    LGBMRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'rmse',
        random_state = 0
    ),
    CatBoostRegressor(
        n_estimators = 100,
        loss_function = 'RMSE',
        learning_rate = 0.1,
        random_state = 0
    )
]
mlf_boost = MLForecast(models = models_boost, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

# pandas data is required by CatBoost and others
y_base_df_pd = y_base_df.to_pandas()

cv_res_boost_base = calibrate_evaluate_plot(
    mlf_boost, df = y_base_df_pd, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_boost_base['cv_results']
cv_res_boost_base['accuracy_table']
cv_res_boost_base['plot'].show()

cv_res_boost_wave = calibrate_evaluate_plot(
    mlf_boost, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_boost_wave['cv_results']
cv_res_boost_wave['accuracy_table']
cv_res_boost_wave['plot'].show()

cv_res_boost_lag = calibrate_evaluate_plot(
    mlf_boost, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_boost_lag['cv_results']
cv_res_boost_lag['accuracy_table']
cv_res_boost_lag['plot'].show()



# CUBIST ------------------------------------------------------------------

# - Like XGBoost, but the terminal (final) nodes are fit using linear regression
# - Does better than tree-based algorithms when time series has trend
# - Can predict beyond maximum
from cubist import Cubist


# * Engines ---------------------------------------------------------------

models_cub = [
    Cubist(
        n_rules = 100, 
        n_committees = 10,
        neighbors = 7,
        random_state = 0
    )
]
mlf_cub = MLForecast(models = models_cub, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_cub_base = calibrate_evaluate_plot(
    mlf_cub, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_cub_base['cv_results']
cv_res_cub_base['accuracy_table']
cv_res_cub_base['plot'].show()

cv_res_cub_wave = calibrate_evaluate_plot(
    mlf_cub, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_cub_wave['cv_results']
cv_res_cub_wave['accuracy_table']
cv_res_cub_wave['plot'].show()

cv_res_cub_lag = calibrate_evaluate_plot(
    mlf_cub, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_cub_lag['cv_results']
cv_res_cub_lag['accuracy_table']
cv_res_cub_lag['plot'].show()



# NEURAL NETWORK ----------------------------------------------------------

# - Single Layer Multi-layer Perceptron Network
# - Simple network - Like linear regression with non-linear functions
# - Can improve learning by adding more hidden units, epochs, etc
from sklearn.neural_network import MLPRegressor


# * Engines ---------------------------------------------------------------

models_nnet = [
    MLPRegressor(
        hidden_layer_sizes = (10,),
        learning_rate_init = 0.1,
        alpha = 0.1,
        activation = 'relu',
        solver = 'adam',
        random_state = 0
    )
]
mlf_nnet = MLForecast(models = models_nnet, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

cv_res_nnet_base = calibrate_evaluate_plot(
    mlf_nnet, df = y_base_df, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_nnet_base['cv_results']
cv_res_nnet_base['accuracy_table']
cv_res_nnet_base['plot'].show()

cv_res_nnet_wave = calibrate_evaluate_plot(
    mlf_nnet, df = y_wave_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_nnet_wave['cv_results']
cv_res_nnet_wave['accuracy_table']
cv_res_nnet_wave['plot'].show()

cv_res_nnet_lag = calibrate_evaluate_plot(
    mlf_nnet, df = y_lag_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 3  
)
cv_res_nnet_lag['cv_results']
cv_res_nnet_lag['accuracy_table']
cv_res_nnet_lag['plot'].show()



# ML Models' Performance Comparison ---------------------------------------

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from cubist import Cubist
from sklearn.neural_network import MLPRegressor

# * Engines ---------------------------------------------------------------

models_ts = [
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
    LGBMRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'rmse',
        random_state = 0
    ),
    CatBoostRegressor(
        n_estimators = 100,
        loss_function = 'RMSE',
        learning_rate = 0.1,
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
mlf_ts = MLForecast(models = models_ts, freq = '1d', num_threads = 1)


# * Evaluation ------------------------------------------------------------

# only base recipe for simplicity
# pandas conversion necessary for CatBoost and other boosting models
y_base_df_pd = y_base_df.to_pandas()

cv_res_ts = calibrate_evaluate_plot(
    mlf_ts, df = y_base_df_pd, h = horizon, 
    prediction_intervals = intervals, level = levels,
    max_insample_length = horizon * 3  
)
cv_res_ts['cv_results']
print_accuracy_table(cv_res_ts['accuracy_table'])
cv_res_ts['plot'].show()


# * Refitting & Forecasting -----------------------------------------------

# pandas conversion necessary for CatBoost and other boosting models
y_base_df_pd = y_base_df.to_pandas()
forecast_y_base_df_pd = forecast_y_base_df.to_pandas()

fit_ts = mlf_ts.fit(df = y_base_df_pd, prediction_intervals = intervals, static_features = [])
preds_df_ts = fit_ts.predict(h = horizon, level = levels, X_df = forecast_y_base_df_pd)

plot_series(y_base_df_pd, preds_df_ts, max_insample_length = horizon * 2, engine = 'plotly').show()


# * Select Best Model -----------------------------------------------------

preds_best_df = get_best_model_forecast(preds_df_ts, cv_res_ts['accuracy_table'], 'rmse')
plot_series(
    y_base_df_pd, preds_best_df, level = levels,
    max_insample_length = horizon * 2, engine = 'plotly'
).show()


# * Back-transform --------------------------------------------------------

back_df = back_transform_data(y_base_df_pd, params)
back_fcst_best_df = back_transform_forecasts(preds_best_df, params)
plot_series(
    back_df, back_fcst_best_df, level = levels,
    max_insample_length = horizon * 3, engine = 'plotly'
).show()
