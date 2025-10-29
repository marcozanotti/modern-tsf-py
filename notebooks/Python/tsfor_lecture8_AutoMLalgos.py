# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 8: Automatic Machine Learning Algorithms ------------------------
# Marco Zanotti

# Goals:
# - Nixtla Auto-models
# - H20
# - AutoGluon



# Packages ----------------------------------------------------------------

import os
import pickle
import re

import numpy as np
import pandas as pd
# import polars as pl
import pytimetk as tk

from statsforecast.utils import ConformalIntervals
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss, MSE
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
intervals = ConformalIntervals(h = horizon, n_windows = 2)

# * Cross-validation Plan -------------------------------------------------

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)

# * External Regressors ---------------------------------------------------

y_df = data_prep_df.select_columns()
y_xregs_df = data_prep_df.select_columns('(event)')



# Nixtla Auto-models ------------------------------------------------------

from statsforecast import StatsForecast
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from ray.tune.search.hyperopt import HyperOptSearch
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Statistical Models
# Automatic forecasts of large numbers of univariate time series are 
# often needed. It is common to have multiple product lines or skus that 
# need forecasting. In these circumstances, an automatic forecasting 
# algorithm is an essential tool. Automatic forecasting algorithms must
# determine an appropriate time series model, estimate the parameters and 
# compute the forecasts. They must be robust to unusual time series 
# patterns, and applicable to large numbers of series without user 
# intervention.
# With Statsforecast, the Auto version of some well-known statistical 
# models are available to greatly simplify the process of searching for
# the best possible model configuration. Usually the choice of the best 
# hyperparameters is done via Information Criteria (AIC). 
from statsforecast.models import (
    AutoARIMA, 
    AutoETS,
    AutoCES,  
    AutoTheta, 
    AutoTBATS, 
    AutoMFLES
)

# Machine Learning Models
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from mlforecast.auto import (
    AutoMLForecast,
    AutoModel,
    AutoElasticNet, 
    AutoLightGBM
)

# Deep Learning Models
# Deep-learning models are the state-of-the-art in time series forecasting. 
# They have outperformed statistical and tree-based approaches in recent 
# large-scale competitions, such as the M series, and are being increasingly 
# adopted in industry. However, their performance is greatly affected by 
# the choice of hyperparameters. Selecting the optimal configuration, a 
# process called hyperparameter tuning, is essential to achieve the best 
# performance.
# The main steps of hyperparameter tuning are:
# - Define training and validation sets.
# - Define search space.
# - Sample configurations with a search algorithm, train models, and evaluate 
#   them on the validation set.
# - Select and store the best model.
# With Neuralforecast, we automatize and simplify the hyperparameter tuning 
# process with the Auto models. Every model in the library has an Auto version 
# (for example, AutoNHITS, AutoTFT) which can perform automatic hyperparameter
#  selection on default or user-defined search space.
# The Auto models can be used with two backends: Ray’s Tune library and Optuna, 
# with a user-friendly and simplified API, with most of their capabilities.
from neuralforecast.auto import (
    AutoRNN, 
    AutoNHITS
)

# * Engines ---------------------------------------------------------------

# Statistical Models
models_sf = [
    AutoARIMA(season_length = 7),
    AutoETS(season_length = 7),
    AutoCES(season_length = 7),
    AutoTheta(season_length = 7),
    AutoTBATS(season_length = [7, 14, 30]),
    AutoMFLES(test_size = horizon, n_windows = 1, season_length = 7, metric = 'mse')
]
sf = StatsForecast(
    models = models_sf, 
    freq = 'D', 
    n_jobs = -1,
)

# Machine Learning Models
models_mlf = [
    AutoElasticNet(),
    AutoLightGBM()
]
mlf = AutoMLForecast(
    models = models_mlf,
    freq = 'D',
    season_length = 7, 
    num_threads = -1
)

# Deep Learning Models
rnn_config = AutoRNN.get_default_config(h = horizon, backend = 'ray')
rnn_config
rnn_config['futr_exog_list'] = ['event']
rnn_config['hist_exog_list'] = ['event']
# rnn_config['max_steps'] = 10 # set to show quick results

nhits_config = AutoNHITS.get_default_config(h = horizon, backend = 'ray') 
nhits_config
nhits_config['futr_exog_list'] = ['event']
nhits_config['hist_exog_list'] = ['event']
# nhits_config['max_steps'] = 10 # set to show quick results

models_nf = [
    AutoRNN(
        h = horizon,
        loss = MQLoss(level = levels),
        config = rnn_config,
        search_alg = HyperOptSearch(),
        backend = 'ray',
        num_samples = 10,
        cpus = 12
    ),
    AutoNHITS(
        h = horizon,
        loss = MQLoss(level = levels),
        config = nhits_config,
        search_alg = HyperOptSearch(),
        backend = 'ray',
        num_samples = 10,
        cpus = 12
    )
]
nf = NeuralForecast(
    models = models_nf,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

# Statistical Models
cv_res_sf = pex.calibrate_evaluate_plot(
    object = sf, data = y_xregs_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_sf['cv_results']
cv_res_sf['accuracy_table']
cv_res_sf['plot'].show()

# Machine Learning Models
# To see the results use .fit and then use .predict
# to produce the forecasts with the optimal models
mlf.fit(
    df = y_xregs_df, h = horizon, n_windows = 1,
    prediction_intervals = intervals,
    num_samples = 2 # number of trials to run
)
mlf.results_['AutoElasticNet']
mlf.models_
mlf.results_['AutoLightGBM'].best_params
mlf_preds_df = mlf.predict(h = horizon, level = levels)
mlf_preds_df

# Deep Learning Models
cv_res_nf = pex.calibrate_evaluate_plot(
    object = nf, data = y_xregs_df, 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_nf['cv_results']
cv_res_nf['accuracy_table']
cv_res_nf['plot'].show()

# To see the results use .fit and then use .predict
# to produce the forecasts with the optimal models
nf.fit(df = y_xregs_df, val_size = horizon)
nf.models[0].results.get_dataframe()
nf.models[1].results.get_dataframe()
nf_preds_df = nf.predict(futr_df = forecast_df)
nf_preds_df



# H2O - Automatic ML Framework --------------------------------------------

# H2O AI
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html
# Algorithms
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html

import h2o
from h2o.automl import H2OAutoML

# * Initialize H2O --------------------------------------------------------

# Dependency on JAVA
# Possible problems related to initialization of H2O from R / Python API:
# - Old JAVA version
# - root privileges
# - JAVA 32bit installed and not 64bit
# - JAVA_HOME env variable not set, Sys.getenv('JAVA_HOME')
# Solutions:
# - https://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/index.html
# - https://docs.h2o.ai/h2o/latest-stable/h2o-docs/faq/java.html
# - https://stackoverflow.com/questions/3892510/set-environment-variables-for-system-in-r
# - Sys.setenv('JAVA_HOME')

# Common steps:
# 1) Uninstall H2O
# if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
# if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# 2) Install the latest version of JAVA
# https://www.oracle.com/technetwork/java/javase/downloads/index.html
# 3) Install H2O again
# install.packages("h2o", type = "source", repos = (c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
# library(h2o)
# 4) Set the JAVA_HOME env variable
# Sys.getenv('JAVA_HOME')
# Sys.setenv(JAVA_HOME="/usr/lib/jvm/jdk-17/")
# Sys.getenv('JAVA_HOME')

h2o.init(nthreads = -1)

# * Format data -----------------------------------------------------------

# select variables
y_xregs_df_h2o = data_prep_df \
    .select_columns('(_lag_)|(event)|(holiday)|(_quarter)|(_month)|(_wday)') \
    .dropna()
y_xregs_df_h2o
forecast_df_h2o = forecast_df \
    .select_columns('(_lag_)|(event)|(holiday)|(_quarter)|(_month)|(_wday)') \
    .drop('y', axis = 1)
forecast_df_h2o

# create train / test split
train_df = y_xregs_df_h2o.head(n = -horizon)
test_df = y_xregs_df_h2o.tail(n = horizon)

# convert data.frame into h2o.frame
train_hf = h2o.H2OFrame(train_df)
test_hf = h2o.H2OFrame(test_df)
forecast_hf = h2o.H2OFrame(forecast_df_h2o)

# identify predictors and response
X = train_hf.columns
X.remove('unique_id')
X.remove('ds')
X.remove('y')
X
y = 'y'

# * Engines ---------------------------------------------------------------

# Algorithms:
# - DRF (This includes both the Distributed Random Forest (DRF) and
#   Extremely Randomized Trees (XRT) models.
# - GLM (Generalized Linear Model with regularization)
# - XGBoost (XGBoost GBM)
# - GBM (H2O GBM)
# - DeepLearning (Fully-connected multi-layer artificial neural network)
# - StackedEnsemble (Stacked Ensembles, includes an ensemble of all the
#   base models and ensembles using subsets of the base models)

# AutoML for 20 base models
aml = H2OAutoML(
    max_runtime_secs = 120,
    # max_runtime_secs_per_model = 30,
    max_models = 30, 
    nfolds = -1,
    sort_metric = 'RMSE',
    # include_algos = c("DRF"),
    # exclude_algos = c("DeepLearning"), # remove deeplearning for computation time
    project_name = 'tsf_test',
    seed = 1
)

# * Evaluation ------------------------------------------------------------

# fitting
aml.train(x = X, y = y, training_frame = train_hf, leaderboard_frame = test_hf)

# view the AutoML Leaderboard
lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
lb

# extract leader model
leader = aml.leader
# aml.get_best_model(criterion = 'mae')
# aml.get_best_model(algorithm = 'xgboost')
# aml.get_best_model(algorithm = 'xgboost', criterion = 'mae')
# h2o.get_model('XRT_1_AutoML_1_20241009_103759') 

# predict
preds = leader.predict(test_hf)
preds

# evaluate
performance = leader.model_performance(test_hf)
performance

# plot
preds_df = pd.concat(
    [
        test_df.select_columns().reset_index(drop = True),
        preds.as_data_frame().rename(columns = {'predict': 'H2O_AutoML'})
    ], 
    axis = 1
)
plot_series(forecasts_df = preds_df, engine = 'plotly').show()

# * Refitting & Forecasting -----------------------------------------------

# leader.predict(forecast_hf)

# to refit the best model, it is necessary to extract the model parameters
# and fit it manually with h2o specific models (not AutoML)
leader.params.keys()
leader.params



# Stop the H20 cluster !!!!!!!!!!!!!!!!!!!!!!!!!!1
h2o.shutdown()

