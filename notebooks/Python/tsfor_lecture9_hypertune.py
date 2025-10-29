# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 9: Hyperparameter Tuning ----------------------------------------
# Marco Zanotti

# Goals:
# - Sequential / Non-Sequential Models
# - Time Series / V-Fold Cross Validation
# - Grid Search / Random Search / Bayesian Optimization
# - Tuning with  ML and DL models

# Challenges:
# - Challenge 3 - Hyperparameter Tuning



# Packages ----------------------------------------------------------------

import os
import pickle
import re

import numpy as np
import pandas as pd

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean, ExponentiallyWeightedMean, ExpandingMean
)
from mlforecast.utils import PredictionIntervals
from utilsforecast.evaluation import evaluate
from neuralforecast.losses.pytorch import MQLoss
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
intervals = PredictionIntervals(h = horizon, n_windows = 2, method = 'conformal_distribution')


# * Train / Test Split ----------------------------------------------------

train_df = data_prep_df \
    .select_columns('(event)|(holiday)|(inter_)|(ds_)') \
    .head(n = -horizon)
test_df = data_prep_df \
    .select_columns('(event)|(holiday)|(inter_)|(ds_)') \
    .tail(n = horizon)
fcst_df = forecast_df \
    .select_columns('(event)|(holiday)|(inter_)|(ds_)')

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)



# Model Types -------------------------------------------------------------

# Sequential Models Tuning
# - Sequential Model Definition:
#   - Creates Lags internally
#   - Predicts next H observations (recursive)
#   - All data must be sequential (ordered by date)
#   - Cannot use V-Fold Cross Validation / Must use Time Series Cross Validation
# - Examples of Sequential Models:
#   - ARIMA
#   - Exponential Smoothing
#   - NNETAR

# Non-Sequential Models Tuning
# - Non-Sequential Model Definition:
#   - Uses date features
#   - Lags Created Externally
#   - Spline can be modeled with random missing observations
#   - Can be Tuned using K-Fold Cross Validation
# - Examples:
#   - Machine Learning Algorithms that use Calendar Features (e.g. GLMNet, XGBoost)
#   - Prophet
# - IMPORTANT: Can use time_series_cv() or vfold_cv(). 
#   Usually better performance with vfold_cv().



# Validation Strategies ---------------------------------------------------

# Time-series cross-validation
pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 12, step_size = 7, 
    engine = 'matplotlib'
)

# V-fold cross-validation



# Search Types ------------------------------------------------------------

# Grid Search
# Grid search is one of the most straightforward methods for hyperparameter 
# tuning. It systematically works through multiple combinations of parameter 
# options, evaluating each one to determine the best performance. While 
# this method guarantees finding the optimal parameters, it can be 
# computationally expensive and time-consuming, especially with a large 
# number of hyperparameters.

# Advantages of Grid Search
# - Exhaustive Search: Evaluates all possible combinations, ensuring the
#   best model is found.
# - Simplicity: Easy to implement and understand.
# Disadvantages of Grid Search
# - Time-Consuming: Can take a long time to complete, particularly with 
#   a large parameter space.
# - Scalability Issues: As the number of hyperparameters increases, the 
#   time required grows exponentially.

# Random Search
# Random search addresses some of the limitations of grid search by 
# randomly sampling from the hyperparameter space. This method can often 
# yield better results in less time, as it does not evaluate every combination.

# Benefits of Random Search
# - Efficiency: Can find good hyperparameters faster than grid search.
# - Flexibility: Allows for exploration of a wider range of values.

# Bayesian Optimization
# Bayesian optimization is a more sophisticated approach that builds 
# a probabilistic model of the function mapping hyperparameters to a 
# target objective. It uses this model to select the most promising 
# hyperparameters to evaluate next, balancing exploration and exploitation.

# Key Features of Bayesian Optimization
# - Model-Based: Utilizes a surrogate model to predict performance.
# - Adaptive Sampling: Focuses on areas of the hyperparameter space that 
#   are likely to yield better results.



# Tuning - ML Models ------------------------------------------------------

from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor

import optuna
from mlforecast.auto import (
    AutoMLForecast,
    AutoModel,
    AutoLightGBM,
    AutoElasticNet
)
optuna.logging.set_verbosity(optuna.logging.ERROR)

# * Non-Optimized Models --------------------------------------------------
models_mlf = [
    ElasticNet(l1_ratio = 0.5),
    LGBMRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'rmse',
        random_state = 0
    )  
]

mlf = MLForecast(
    models = models_mlf,
    freq = 'D', 
    num_threads = 1,
    lags = [1, 2, 7, 14, 30],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        30: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

mlf_fit = mlf.fit(
    df = train_df, 
    prediction_intervals = intervals,
    static_features = []
)
mlf_preds_df = mlf_fit.predict(
    h = horizon, 
    level = levels, 
    X_df = test_df.drop('y', axis = 1)
)

evaluate(
    df = mlf_preds_df \
        .merge(test_df.select_columns(), on = ['unique_id', 'ds']), 
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)
plot_series(
    pd.concat([train_df, test_df]), mlf_preds_df,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()


# * Optimized Models ------------------------------------------------------

# ** Default optimization -------------------------------------------------

# We have default search spaces for some models and we can define default 
# features to look for based on the length of the seasonal period of your 
# data. You just need to use the Auto-model couterpart.

# ** Tuning model parameters ----------------------------------------------

# Otherwise, you can provide your own model with its search space to perform 
# the optimization. The search space should be a function that takes an 
# optuna trial and returns the model parameters. Then simply create the auto
# version of the desired model using the AutoModel function specifing 
# the model and the configuration.
def lgb_config(trial: optuna.Trial):
    return {
        'learning_rate': 0.05,
        'verbosity': -1,
        'num_leaves': trial.suggest_int('num_leaves', 2, 128, log = True),
        'objective': trial.suggest_categorical('objective', ['l1', 'l2', 'rmse']),
    }
my_lgb = AutoModel(
    model = LGBMRegressor(),
    config = lgb_config,
)

# ** Tuning features ------------------------------------------------------

# The MLForecast class defines the features to build in its constructor. 
# You can tune the features by providing a function through the init_config 
# argument, which will take an optuna trial and produce a configuration to 
# pass to the MLForecast constructor. 
def init_config(trial: optuna.Trial):
    # lag_transforms = [
    #     ExponentiallyWeightedMean(alpha=0.3),
    #     RollingMean(window_size=24 * 7, min_samples=1),
    # ]
    # lag_to_transform = trial.suggest_categorical('lag_to_transform', [24, 48])
    lags = [7, 14, 30]
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        30: [
            RollingMean(window_size = 30),
            # RollingMean(window_size = 60), # too few data points for this transformation 
            # RollingMean(window_size = 90), # too few data points for this transformation
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
    date_features = []
    res = {
        'lags': lags, 
        'lag_transforms': lag_transforms,
        'date_features': date_features
    }
    return res 

# ** Tuning fit parameters ------------------------------------------------

# The MLForecast.fit method takes some arguments that could improve the 
# forecasting performance of your models, such as dropna and static_features. 
# If you want to tune those you can provide a function to the fit_config argument.
def fit_config(trial: optuna.Trial):
    # if trial.suggest_int('use_id', 0, 1):
    #     static_features = ['unique_id']
    # else:
    #     static_features = None
    static_features = []
    prediction_intervals = PredictionIntervals(
        h = horizon, n_windows = 2, method = 'conformal_distribution'
    )
    dropna = True
    res = {
        'static_features': static_features, 
        'prediction_intervals': prediction_intervals,
        'dropna': dropna
    }
    return res

# ** Engines --------------------------------------------------------------
models_auto_mlf = {
    'elanet_default': AutoElasticNet(), 
    'lgbm_default': AutoLightGBM(),
    'my_lgbm': my_lgb
}
auto_mlf = AutoMLForecast(
    models = models_auto_mlf,
    freq = 'D',
    init_config = init_config,
    fit_config = fit_config,
    # season_length = 7, # not to use with init_config
)

# ** Tuning ---------------------------------------------------------------

# Finally you can run the optimization process through the .fit method.
auto_mlf.fit(
    train_df,
    h = horizon,
    n_windows = 6, # 12 is not possible because too short time series (also for Conformal intervals)
    refit = 7, # step_size
    num_samples = 20 # number of trials to run
)

# ** Extracting results ---------------------------------------------------

# There is one optimization process per model. This is because 
# different models can make use of different features. So after 
# the optimization process is done for each model the best 
# configuration is used to retrain the model using all of the data. 
# These final models are MLForecast objects and are saved in the 
# models_ attribute.
auto_mlf.results_['elanet_default'].best_params
auto_mlf.results_['lgbm_default'].best_params
auto_mlf.results_['my_lgbm'].best_params

# ** Evaluation -----------------------------------------------------------

auto_mlf_preds_df = auto_mlf.predict(h = horizon, level = levels, X_df = test_df)
auto_mlf_preds_df

evaluate(
    df = auto_mlf_preds_df \
        .merge(test_df.select_columns(), on = ['unique_id', 'ds']), 
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)
plot_series(
    pd.concat([train_df, test_df]), auto_mlf_preds_df,
    max_insample_length = horizon * 2,
    level = None,
    engine = 'plotly'
).show()

# ** Refitting & Forecasting ----------------------------------------------

# To refit the optimized model you have to save the model object and 
# follow the usual ML workflow using MLForecast().ft and MLForecast.predict().
auto_mlf.models_



# Tuning - DL Models ------------------------------------------------------

# Deep-learning models are the state-of-the-art in time series forecasting. 
# They have outperformed statistical and tree-based approaches in recent 
# large-scale competitions, such as the M series, and are being increasingly
# adopted in industry. However, their performance is greatly affected by the 
# choice of hyperparameters. Selecting the optimal configuration, a process 
# called hyperparameter tuning, is essential to achieve the best performance.
# The main steps of hyperparameter tuning are:
# - Define training and validation sets.
# - Define search space.
# - Sample configurations with a search algorithm, train models, and evaluate 
#   them on the validation set.
# - Select and store the best model.
# With Neuralforecast, we automatize and simplify the hyperparameter tuning 
# process with the Auto models. Every model in the library has an Auto 
# version (for example, AutoNHITS, AutoTFT) which can perform automatic 
# hyperparameter selection on default or user-defined search space.
# The Auto models can be used with two backends: Ray’s Tune library and 
# Optuna, with a user-friendly and simplified API, with most of their 
# capabilities.

# ATTENTION: we are going to use the Ray's backend because we can use more cpus.

from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    MLP,
    GRU,
    NHITS, 
    NBEATSx
) 
from neuralforecast.auto import (
    AutoMLP,
    AutoGRU,
    AutoNBEATSx,
    AutoNHITS
)
optuna.logging.set_verbosity(optuna.logging.ERROR)

# * Non-Optimized Models --------------------------------------------------

models_nf = [
    MLP(
        h = horizon,
        input_size = 30,
        num_layers = 2,
        hidden_size = 128,
        max_steps = 300,
        loss = MQLoss(level = levels),
        futr_exog_list = ['event'],
        hist_exog_list = ['event'],
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
        max_steps = 300,
        loss = MQLoss(level = levels),
        futr_exog_list = ['event'],
        hist_exog_list = ['event'],
        random_seed = 0 
    ),
    NBEATSx(
        h = horizon, 
        input_size = 30,
        stack_types = ['identity', 'trend', 'seasonality'],
        loss = MQLoss(level = levels),
        max_steps = 300,
        futr_exog_list = ['event'],
        hist_exog_list = ['event'],
        random_seed = 0
    ),
    NHITS(
        h = horizon, 
        input_size = 30,
        loss = MQLoss(level = levels),
        max_steps = 300,
        futr_exog_list = ['event'],
        hist_exog_list = ['event'],
        random_seed = 0
    )
]

nf = NeuralForecast(
    models = models_nf,
    freq = 'D'    
)

nf.fit(df = train_df)
nf_preds_df = nf.predict(futr_df = test_df.drop('y', axis = 1)) \
    .rename(columns = lambda x: re.sub('-median', '', x))

evaluate(
    df = nf_preds_df \
        .merge(test_df.select_columns(), on = ['unique_id', 'ds']), 
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)
plot_series(
    pd.concat([train_df, test_df]), nf_preds_df,
    max_insample_length = horizon * 2,
    # level = levels,
    engine = 'plotly'
).show()


# * Optimized Models ------------------------------------------------------

# ** Default optimization -------------------------------------------------

# Each Auto model contains a default search space that was extensively 
# tested on multiple large-scale datasets. Search spaces are specified 
# with a function that returns a dictionary, where keys corresponds to 
# the model’s hyperparameter and the value is a suggest function to specify 
# how the hyperparameter will be sampled. For example, use suggest_int 
# to sample integers uniformly, and suggest_categorical to sample values 
# of a list. 

# The default search space dictionary can be accessed through the 
# get_default_config function of the Auto model. This is useful if you 
# wish to use the default parameter configuration but want to change one 
# or more hyperparameter spaces without changing the other default values.
# To extract the default config, you need to define: 
# - h: forecasting horizon. 
# - backend: backend to use. 
# - n_series: Optional, the number of unique time series, required only 
#   for Multivariate models.

# ATTENTION:
# to get default configuration you need to work with the 'ray' backend
config = AutoMLP.get_default_config(h = horizon, backend = 'ray')
config.keys()
config['hidden_size'].categories
config['num_layers'].lower, config['num_layers'].upper

mlp_config = AutoMLP.get_default_config(h = horizon, backend = 'ray')
gru_config = AutoGRU.get_default_config(h = horizon, backend = 'ray')
nbeats_config = AutoNBEATSx.get_default_config(h = horizon, backend = 'ray')
nhits_config = AutoNHITS.get_default_config(h = horizon, backend = 'ray') 

# ** Tuning model parameters ----------------------------------------------

# Define a custom grid configuration for NHITS
nhits_config_custom = {
    "max_steps": 100, # Number of SGD steps
    "input_size": 24, # Size of input window
    "learning_rate": tune.loguniform(1e-5, 1e-1), # Initial Learning rate
    "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]), # MaxPool's Kernelsize
    "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
    "val_check_steps": 50, # Compute validation every 50 steps
    "random_seed": tune.randint(1, 10), # Random seed
    "futr_exog_list": ['event'],
    "hist_exog_list": ['event']
}

# Add non tunable exogenous features
mlp_config['futr_exog_list'] = ['event']
mlp_config['hist_exog_list'] = ['event']

gru_config['futr_exog_list'] = ['event']
gru_config['hist_exog_list'] = ['event']

nbeats_config['futr_exog_list'] = ['event']
nbeats_config['hist_exog_list'] = ['event']

nhits_config['futr_exog_list'] = ['event']
nhits_config['hist_exog_list'] = ['event']

# to quickly see results
# mlp_config['max_steps'] = 10
# gru_config['max_steps'] = 10
# nbeats_config['max_steps'] = 10
# nhits_config['max_steps'] = 10

# ** Engines --------------------------------------------------------------

models_auto_nf = [
    AutoMLP(
        h = horizon,
        loss = MQLoss(level = levels),
        valid_loss = MQLoss(level = levels),
        config = mlp_config,
        search_alg = HyperOptSearch(),
        backend = 'ray',
        num_samples = 20, # number of models to look for in the search space
        cpus = 12 # only available with Ray's backend
    ),
    AutoGRU(
        h = horizon,
        loss = MQLoss(level = levels),
        valid_loss = MQLoss(level = levels),
        config = gru_config,
        search_alg = HyperOptSearch(),
        backend = 'ray',
        num_samples = 20, # number of models to look for in the search space
        cpus = 12 # only available with Ray's backend
    ),
    AutoNBEATSx(
        h = horizon,
        loss = MQLoss(level = levels),
        valid_loss = MQLoss(level = levels),
        config = nbeats_config,
        search_alg = HyperOptSearch(),
        backend = 'ray',
        num_samples = 20, # number of models to look for in the search space
        cpus = 12 # only available with Ray's backend
    ),
    AutoNHITS(
        h = horizon,
        loss = MQLoss(level = levels),
        valid_loss = MQLoss(level = levels),
        config = nhits_config,
        search_alg = HyperOptSearch(),
        backend = 'ray',
        num_samples = 20,
        cpus = 12
    ),  
    AutoNHITS(
        h = horizon,
        loss = MQLoss(level = levels),
        valid_loss = MQLoss(level = levels),
        config = nhits_config_custom,
        search_alg = HyperOptSearch(),
        backend = 'ray',
        num_samples = 20,
        cpus = 12,
        alias = 'NHITS_custom'
    )
]
auto_nf = NeuralForecast(
    models = models_auto_nf,
    freq = 'D'
)

# ** Tuning ---------------------------------------------------------------

# Next, we use the Neuralforecast class to train the Auto model. In this 
# step, Auto models will automatically perform hyperparamter tuning 
# training multiple models with different hyperparameters, producing the 
# forecasts on the validation set, and evaluating them. The best 
# configuration is selected based on the error on a validation set. Only 
# the best model is stored and used during inference.
auto_nf.fit(df = train_df, val_size = horizon)

# ** Extracting results ---------------------------------------------------

auto_nf.models
auto_nf.models[0].results.get_dataframe()
auto_nf.models[1].results.get_dataframe()
auto_nf.models[2].results.get_dataframe()
auto_nf.models[3].results.get_dataframe()
auto_nf.models[4].results.get_dataframe()

# ** Evaluation -----------------------------------------------------------

auto_nf_preds_df = auto_nf.predict(futr_df = test_df) \
    .rename(columns = lambda x: re.sub('-median', '', x))
auto_nf_preds_df

evaluate(
    df = auto_nf_preds_df \
        .merge(test_df.select_columns(), on = ['unique_id', 'ds']), 
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)
plot_series(
    pd.concat([train_df, test_df]), auto_nf_preds_df,
    max_insample_length = horizon * 2,
    level = None,
    engine = 'plotly'
).show()

# ** Refitting & Forecasting ----------------------------------------------

# To refit the optimized model you have to save the model object and 
# follow the usual NF workflow using NeuralForecast().ft and 
# NeuralForecast().predict().
gru_config
auto_nf.models[1].results.get_best_result()
auto_nf.models[1].results \
    .get_dataframe() \
    .sort_values(by = 'loss') \
    .select_columns('(loss)|(config)')

best_auto_model = [
    GRU(
        h = horizon,
        loss = MQLoss(level = levels),
        input_size = 896,
        learning_rate = 0.000364,
        batch_size = 32,
        context_size = 10,
        encoder_hidden_size = 50,
        encoder_n_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 500,
        futr_exog_list = ['event'],
        hist_exog_list = ['event'],
        random_seed = 17       
    )
]
best_auto_nf = NeuralForecast(
    models = best_auto_model,
    freq = 'D' 
)
best_auto_nf.fit(df = pd.concat([train_df, test_df]))
best_auto_nf_fcst_df = best_auto_nf.predict(futr_df = fcst_df) \
    .rename(columns = lambda x: re.sub('-median', '', x))
plot_series(
    pd.concat([train_df, test_df]), best_auto_nf_fcst_df,
    max_insample_length = horizon * 2,
    level = None,
    engine = 'plotly'
).show()
