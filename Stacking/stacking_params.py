from sklearn.metrics import mean_squared_error as mse 
from sklearn.linear_model import SGDRegressor as SGD
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.ensemble import ExtraTreesRegressor 
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2, activity_l2
import keras.layers.advanced_activations as aa
from keras.objectives import poisson
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.svm import LinearSVR 
from sklearn.ensemble import RandomForestRegressor 

DEBUG_MODE = False

if not DEBUG_MODE:
    xgb_tree_param = {
        'task': 'regression',
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eta': 0.01,
        'gamma': 0,
        'min_child_weight': 0,
        'max_depth': 9,
        'subsample': 1.0,
        'colsample_bytree': 0.35,
        'num_round': 1000,
        'silent': 1,
        'seed': 2016,  
        "eval_metric" : "rmse",
        'verbose' : 0,
        'lambda': 0.7,
        'alpha': 0.7,
        'BAG_SIZE' : 1,
    }

    xgb_linear_param = {
        'task': 'regression',
        'booster': 'gblinear',
        'objective': 'reg:linear',
        'eta' : 0.01,
        'lambda' : 0.5,
        'alpha' : 0.2,
        'lambda_bias' : 0.1,
        'num_round' : 600,
        'silent' : 1,
        'seed': 2016,
        "eval_metric" : "rmse",
        'verbose' : 0,
        'BAG_SIZE' : 1,
    }

    ada_param = {
        'n_estimators' : 200,
        'learning_rate' : 1,
        'loss' : 'linear',
        #'n_jobs' : -1,
        'BAG_SIZE' : 1,
    }

    sgd_param = {
        'loss' : 'huber',
        'penalty' : 'l2',
        'alpha' : 0.0001,
        'l1_ratio' : 0.2,
        'fit_intercept' : True,
        'n_iter' : 50,
        'verbose' : 0,
        'BAG_SIZE' : 1,
    }

    et_param = {
        'n_estimators' : 1000,
        'min_samples_split' : 3,
        'max_features' : 0.3,
        'n_jobs' : -1,
        'max_depth' : None,
        'verbose' : 0,
        'BAG_SIZE' : 1,
    }

    keras_param = {
        'BATCH_SIZE' : 512,
        'EPOCHS' : 40,
        'HIDDEN_LAYER_1' : 500, 
        'HIDDEN_LAYER_2' : 128,
        'HIDDEN_DROPOUT_1' : 0.1,
        'HIDDEN_DROPOUT_2' : 0.1,
        'L2_LAMBDA' : 0.00001,
        'BAG_SIZE' : 3,
    }

    knn_param = {
        'weights' : 'distance',
        'algorithm' : 'kd_tree',
        'leaf_size' : 15,
        'n_jobs' : -1,
        'neighbours' : [2**i for i in range(11)],
    }

    lasso_param = {
        'BAG_SIZE' : 1,
        'alpha' : 0.2,
    }

    ridge_param = {
        'BAG_SIZE' : 1,
        'alpha' : 0.2,
    }

    linearSVR_param = {
        'BAG_SIZE':1,
        'C' : 0.01,
        'loss' : 'squared_epsilon_insensitive',
        'epsilon' : 0.05,
        'tol' : 1e-5,
        'dual' : False,
        'max_iter' : 2000,
        'verbose' : 0,
    }

    rf_param = {
        'BAG_SIZE' : 1,
        'n_estimators' : 1000,  
        'min_samples_split' : 3,
        'max_features' : 0.3,
        'n_jobs' : -1,
        'max_depth' : None,
        'verbose' : 0,
    }

    bayes_param = {
        'BAG_SIZE':1,
    }
else:
    xgb_tree_param = {
        'task': 'regression',
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eta': 0.01,
        'gamma': 0,
        'min_child_weight': 0,
        'max_depth': 9,
        'subsample': 1.0,
        'colsample_bytree': 0.30,
        'num_round': 10,
        'silent': 1,
        'seed': 2016,  
        "eval_metric" : "rmse",
        'verbose' : 0,
        'lambda': 0.7,
        'alpha': 0.7,
        'BAG_SIZE' : 1,
    }

    xgb_linear_param = {
        'task': 'regression',
        'booster': 'gblinear',
        'objective': 'reg:linear',
        'eta' : 0.01,
        'lambda' : 0.5,
        'alpha' : 0.2,
        'lambda_bias' : 0.1,
        'num_round' : 10,
        'silent' : 1,
        'seed': 2016,
        "eval_metric" : "rmse",
        'verbose' : 0,
        'BAG_SIZE' : 1,
    }

    ada_param = {
        'n_estimators' : 1,
        'learning_rate' : 1,
        'loss' : 'linear',
        'n_jobs' : -1,
        'BAG_SIZE' : 1,
    }

    sgd_param = {
        'loss' : 'huber',
        'penalty' : 'l2',
        'alpha' : 0.0001,
        'l1_ratio' : 0.2,
        'fit_intercept' : True,
        'n_iter' : 10,
        'verbose' : 0,
        'BAG_SIZE' : 1,
    }

    et_param = {
        'n_estimators' : 5,
        'min_samples_split' : 3,
        'max_features' : 0.3,
        'n_jobs' : -1,
        'max_depth' : None,
        'verbose' : 0,
        'BAG_SIZE' : 1,
    }

    keras_param = {
        'BATCH_SIZE' : 512,
        'EPOCHS' : 10,
        'HIDDEN_LAYER_1' : 500, 
        'HIDDEN_LAYER_2' : 128,
        'HIDDEN_DROPOUT_1' : 0.1,
        'HIDDEN_DROPOUT_2' : 0.1,
        'L2_LAMBDA' : 0.00001,
        'BAG_SIZE' : 1,
    }

    knn_param = {
        'weights' : 'distance',
        'algorithm' : 'kd_tree',
        'leaf_size' : 15,
        'n_jobs' : -1,
        'neighbours' : [1, 2],
    }

    lasso_param = {
        'BAG_SIZE' : 1,
        'alpha' : 0.2,
    }

    ridge_param = {
        'BAG_SIZE' : 1,
        'alpha' : 0.2,
    }

    linearSVR_param = {
        'BAG_SIZE':1,
        'C' : 0.01,
        'loss' : 'squared_epsilon_insensitive',
        'epsilon' : 0.05,
        'tol' : 1e-5,
        'dual' : False,
        'max_iter' : 100,
        'verbose' : 0,
    }

    rf_param = {
        'BAG_SIZE' : 1,
        'n_estimators' : 5,  
        'min_samples_split' : 3,
        'max_features' : 0.3,
        'n_jobs' : -1,
        'max_depth' : None,
        'verbose' : 0,
    }

    bayes_param = {
        'BAG_SIZE':1,
    }















