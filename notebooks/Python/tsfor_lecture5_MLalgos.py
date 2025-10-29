# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 5: Machine Learning Algorithms ----------------------------------
# Marco Zanotti

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

import os
import pickle

import numpy as np
import pandas as pd
import random
# import polars as pl
import pytimetk as tk

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean, SeasonalRollingMean, 
    ExponentiallyWeightedMean, ExpandingMean
)
from mlforecast.target_transforms import LocalBoxCox, LocalStandardScaler
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse
from utilsforecast.plotting import plot_series

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

plot_series(data_prep_df).show()

# * Forecast Horizon ------------------------------------------------------
horizon = 7 * 8 # 8 weeks

# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
# Conformal intervals
# method can be conformal_distribution or conformal_error; 
# conformal_distribution (default) creates forecasts paths based on 
# the cross-validation errors and calculate quantiles using those paths, 
# on the other hand conformal_error calculates the error quantiles to 
# produce prediction intervals. The strategy will adjust the intervals 
# for each horizon step, resulting in different widths for each step. 
# Please note that a minimum of 2 cross-validation windows must be used.
intervals = ConformalIntervals(h = horizon, n_windows = 2, method = 'conformal_distribution')
# or
# intervals = PredictionIntervals(h = horizon, n_windows = 2, method = 'conformal_distribution')

# P.S. n_windows*h should be less than the count of data elements in your time series sequence.
# P.S. Also value of n_windows should be atleast 2 or more.

# * Cross-validation Plan -------------------------------------------------

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)

# * External Regressors ---------------------------------------------------

# back-transform the y to use mlforecast full workflow
# but results are not directly comparable to those of TS because 
# the predictions are automatically back-transformed
# y_df = data_prep_df \
#     .select_columns() \
#     .back_transform_data(params = params)
# y_xregs_df = data_prep_df \
#     .select_columns(
#         regex = '(event)|(holiday)|(inter_)|(ds_mweek_2)|(_sin_)|(_cos_)'
#     ) \
#     .back_transform_data(params = params)
y_df = data_prep_df.select_columns()
y_xregs_df = data_prep_df.select_columns('(event)|(holiday)|(inter_)|(ds_)')


# * Transformations -------------------------------------------------------

# Target trasforms
# Log1p transformation
from sklearn.preprocessing import FunctionTransformer
from mlforecast.target_transforms import GlobalSklearnTransformer
Log1p = FunctionTransformer(func = np.log1p, inverse_func = np.expm1)
Log1p = GlobalSklearnTransformer(Log1p)

# Date features
def is_weekend(ds):
    """Date is weekend"""
    return any([ds.dayofweek == 5, ds.dayofweek == 6])



# Linear Regression -------------------------------------------------------

# - Baseline model for ML
from sklearn.linear_model import LinearRegression

# * Engines ---------------------------------------------------------------

models_lr = [
    LinearRegression()
]

# * Recipe ----------------------------------------------------------------

# the advantage of generating lags and lag_transforms through MLForecast
# API is that the model is automatically fitted in a recursive way hence 
# one can use lags of order lower than the forecast horizon.
# Keep in mind that the recursive strategy suffers from error accumulation. 

mlf_lr = MLForecast(
    models = models_lr,
    freq = 'D', 
    num_threads = 1,
    # target_transforms = [Log1p, LocalStandardScaler()], # test with original y
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    },
    # date_features = [
    #     'year', 'quarter', 'month', 'week', 
    #     'day', 'dayofyear', 'dayofweek', is_weekend
    # ]
)
mlf_lr.preprocess(y_xregs_df, static_features = [])

# * Evaluation ------------------------------------------------------------

# test with different feature sets to see the effects of the features
# data_prep_df[feature_sets['base']]
# data_prep_df[feature_sets['spline']]
# data_prep_df[feature_sets['lag']]
# mlf_lr = MLForecast(
#     models = models_lr,
#     freq = 'D', 
#     num_threads = 1
# )
# cv_res_lr = pex.calibrate_evaluate_plot(
#     mlf_lr, data = data_prep_df[feature_sets['lag']].dropna(), 
#     h = horizon, prediction_intervals = intervals, level = levels,
#     engine = 'plotly', max_insample_length = horizon * 2  
# )

cv_res_lr = pex.calibrate_evaluate_plot(
    mlf_lr, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_lr['cv_results']
cv_res_lr['accuracy_table']
cv_res_lr['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

fit_lr = mlf_lr.fit(
    df = y_xregs_df, 
    prediction_intervals = intervals,
    static_features = []
)
fit_lr.models_['LinearRegression'].intercept_
fit_lr.models_['LinearRegression'].coef_

preds_df_lr = fit_lr.predict(
    h = horizon, level = levels, X_df = forecast_df.drop('y', axis = 1)
)
preds_df_lr

plot_series(
    data_prep_df, preds_df_lr,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()



# Elastic Net -------------------------------------------------------

# - Strengths: Very good for trend
# - Weaknesses: Not as good for complex patterns (i.e. seasonality)
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# * Engines ---------------------------------------------------------------

models_elanet = [
    Ridge(),
    Lasso(),
    ElasticNet(l1_ratio = 0.5)    
]

# * Recipe ----------------------------------------------------------------

mlf_elanet = MLForecast(
    models = models_elanet,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_elanet = pex.calibrate_evaluate_plot(
    mlf_elanet, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_elanet['cv_results']
cv_res_elanet['accuracy_table']
cv_res_elanet['plot'].show()



# MARS --------------------------------------------------------------------

# Multiple Adaptive Regression Splines

# - Strengths: Best algorithm for modeling trend
# - Weaknesses:
#   - Not good for complex patterns (i.e. seasonality)
#   - Don't combine with splines! MARS makes splines.

# pip install sklearn-contrib-py-earth
from pyearth import Earth

# * Engines ---------------------------------------------------------------

models_mars = [
    Earth()
]

# * Recipe ----------------------------------------------------------------

mlf_mars = MLForecast(
    models = models_mars,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_mars = pex.calibrate_evaluate_plot(
    mlf_mars, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_mars['cv_results']
cv_res_mars['accuracy_table']
cv_res_mars['plot'].show()



# SVM ---------------------------------------------------------------------

# Support Vector Machines

# Strengths: Well-rounded algorithm
# Weaknesses: Needs tuned or can overfit and can be computationally inefficient
# - Strengths: Very good for trend
# - Weaknesses: Not as good for complex patterns (i.e. seasonality)
from sklearn.svm import SVR

# * Engines ---------------------------------------------------------------

models_svm = [
    # SVR(kernel = 'linear'), # too slow !!!!!!!!!!!!!!!!!!!!
    # SVR(kernel = 'poly'), # too slow !!!!!!!!!!!!!!!!!!!!
    SVR(kernel = 'rbf')
]

# * Recipe ----------------------------------------------------------------

mlf_svm = MLForecast(
    models = models_svm,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_svm = pex.calibrate_evaluate_plot(
    mlf_svm, data = y_xregs_df,
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_svm['cv_results']
cv_res_svm['accuracy_table']
cv_res_svm['plot'].show()



# KNN ---------------------------------------------------------------------

# K Neighrest Neighbors

# - Strengths: Uses neighboring points to estimate
# - Weaknesses: Cannot predict beyond the maximum/minimum target (e.g. increasing trend)
# - Solution: Model trend separately (if needed).
#   - Can combine with ARIMA, Linear Regression, Mars, or Prophet
from sklearn.neighbors import KNeighborsRegressor

# * Engines ---------------------------------------------------------------

models_knn = [
    KNeighborsRegressor(n_neighbors = 15)
]

# * Recipe ----------------------------------------------------------------

mlf_knn = MLForecast(
    models = models_knn,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_knn = pex.calibrate_evaluate_plot(
    mlf_knn, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_knn['cv_results']
cv_res_knn['accuracy_table']
cv_res_knn['plot'].show()



# GAUSSIAN PROCESS REGRESSION ---------------------------------------------

from sklearn.gaussian_process import GaussianProcessRegressor

# * Engines ---------------------------------------------------------------

models_gp = [
    GaussianProcessRegressor()
]

# * Recipe ----------------------------------------------------------------

mlf_gp = MLForecast(
    models = models_gp,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_gp = pex.calibrate_evaluate_plot(
    mlf_gp, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_gp['cv_results']
cv_res_gp['accuracy_table']
cv_res_gp['plot'].show()



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

# * Recipe ----------------------------------------------------------------

mlf_tree = MLForecast(
    models = models_tree,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_tree = pex.calibrate_evaluate_plot(
    mlf_tree, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_tree['cv_results']
cv_res_tree['accuracy_table']
cv_res_tree['plot'].show()



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
        max_samples = 1,
        max_features = 1,
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

# * Recipe ----------------------------------------------------------------

mlf_rf = MLForecast(
    models = models_rf,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_rf = pex.calibrate_evaluate_plot(
    mlf_rf, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_rf['cv_results']
cv_res_rf['accuracy_table']
cv_res_rf['plot'].show()



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

# * Recipe ----------------------------------------------------------------

mlf_boost = MLForecast(
    models = models_boost,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_boost = pex.calibrate_evaluate_plot(
    mlf_boost, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_boost['cv_results']
cv_res_boost['accuracy_table']
cv_res_boost['plot'].show()



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

# * Recipe ----------------------------------------------------------------

mlf_cub = MLForecast(
    models = models_cub,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_cub = pex.calibrate_evaluate_plot(
    mlf_cub, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_cub['cv_results']
cv_res_cub['accuracy_table']
cv_res_cub['plot'].show()



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

# * Recipe ----------------------------------------------------------------

mlf_nnet = MLForecast(
    models = models_nnet,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_nnet = pex.calibrate_evaluate_plot(
    mlf_nnet, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_nnet['cv_results']
cv_res_nnet['accuracy_table']
cv_res_nnet['plot'].show()



# ML Models' Performance Comparison ---------------------------------------

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from cubist import Cubist

# * Engines ---------------------------------------------------------------

models_ts = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    SVR(kernel = 'rbf'),
    DecisionTreeRegressor(random_state = 0),
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
    )
]
mlf_ts = MLForecast(
    models = models_ts,
    freq = 'D', 
    num_threads = 1,
    lags = [7, 14, 30, 56],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        56: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

# * Evaluation ------------------------------------------------------------

cv_res_ts = pex.calibrate_evaluate_plot(
    object = mlf_ts, data = y_xregs_df,
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_ts['cv_results']
cv_res_ts['accuracy_table'].print_accuracy_table('min')
cv_res_ts['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

fit_ts = mlf_ts.fit(
    df = y_xregs_df, 
    prediction_intervals = intervals,
    static_features = []
)
preds_df_ts = fit_ts.predict(
    h = horizon, level = levels, X_df = forecast_df.drop('y', axis = 1)
)

plot_series(
    y_xregs_df, preds_df_ts,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()

# * Select Best Model -----------------------------------------------------

preds_best_df = preds_df_ts \
    .get_best_model_forecast(cv_res_ts['accuracy_table'], 'rmse')
plot_series(
    y_xregs_df, preds_best_df,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()

# * Back-transform --------------------------------------------------------

data_back_dict = pex.back_transform_data(y_xregs_df, params, preds_df_ts)
plot_series(
    data_back_dict['data_back'], 
    data_back_dict['forecasts_back'], 
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly', 
).show()





# XX ----------------------------------------------------------------------

# from sklearn.linear_model import Lasso, Ridge, ElasticNet

# # * Engines ---------------------------------------------------------------

# help(ElasticNet)
# models_xx = [

# ]

# # * Recipe ----------------------------------------------------------------

# mlf_xx = MLForecast(
#     models = models_xx,
#     freq = 'D', 
#     num_threads = 1,
#     lags = [7, 14, 30, 56],
#     lag_transforms = {
#         7: [
#             RollingMean(window_size = 7),
#             ExpandingMean()
#         ],
#         14: [
#             RollingMean(window_size = 14)            
#         ],
#         56: [
#             RollingMean(window_size = 30),
#             RollingMean(window_size = 60), 
#             RollingMean(window_size = 90),
#             ExponentiallyWeightedMean(alpha = 0.3)
#         ]
#     }
# )

# # * Evaluation ------------------------------------------------------------

# cv_res_xx = pex.calibrate_evaluate_plot(
#     mlf_xx, data = y_xregs_df, 
#     h = horizon, prediction_intervals = intervals, level = levels,
#     engine = 'plotly', max_insample_length = horizon * 2  
# )
# cv_res_xx['cv_results']
# cv_res_xx['accuracy_table']
# cv_res_xx['plot'].show()