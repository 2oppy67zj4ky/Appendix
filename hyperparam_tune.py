from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, randint
import os
import sys
file_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, str(file_path))
from iife import iife
from data import *
from helper import *
import numpy as np
import json
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.model_selection import KFold
import time
import ray
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from knncmi import *
import pickle
import lightgbm as lgb
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def hyperparam_tune(task, model_type, seed, data=None):
	if task == "openml586":
		X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_openml586(seed)
		scoring=make_scorer(RAE_comp)
	elif task == "cal_housing":
		X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_cal_housing(seed)
		scoring=make_scorer(RAE_comp)
	elif task == "jungle_chess":
		X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_jungle_chess(seed)
		scoring="f1_micro"
	
	if data:
		X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = data

	if (model_type=="LR" or model_type=="lasso"):
		if "cat" in vartype_list:
			enc = OneHotEncoder(handle_unknown='ignore')
			X_train_num = X_train[:,np.where(np.array(vartype_list)=="num")[0]]
			X_train_cat = X_train[:,np.where(np.array(vartype_list)=="cat")[0]]
			X_test_num = X_test[:,np.where(np.array(vartype_list)=="num")[0]]
			X_test_cat = X_test[:,np.where(np.array(vartype_list)=="cat")[0]]
			enc.fit(X_train_cat)
			X_train_ohe = enc.transform(X_train_cat).toarray()
			X_test_ohe = enc.transform(X_test_cat).toarray()

			X_train_temp = np.concatenate((X_train_num,X_train_ohe),axis=1)
			X_test_temp = np.concatenate((X_test_num,X_test_ohe),axis=1)
		else:
			X_train_temp=X_train
			X_test_temp=X_test
		scaler = MinMaxScaler()
		scaler.fit(X_train_temp)
		X_train_temp=scaler.transform(X_train_temp)
		X_test_temp=scaler.transform(X_test_temp)
	else:
		X_train_temp=X_train
		X_test_temp=X_test
	# X_train_temp = np.nan_to_num(X_train_temp,nan=-1, posinf=-1, neginf=-1)
	if model_type=="LR":
		model = LogisticRegression(max_iter=50000)
		distributions = dict(C=loguniform(a=0.00001, b=100))
	elif model_type=="lasso":
		model = Lasso(max_iter=50000)
		distributions = dict(alpha=loguniform(a=0.00001, b=100))
	elif model_type=="RF":
		model=RandomForestClassifier(random_state=0)
		distributions = dict(
			n_estimators=randint(5,250),
			max_depth=randint(1,250),
			max_features=uniform(.01,.99),
			max_samples=uniform(.1,.9)
		)
	elif model_type=="RFR":
		model=RandomForestRegressor(random_state=0)
		distributions = dict(
			n_estimators=randint(5,250),
			max_depth=randint(1,250),
			max_features=uniform(.01,.99),
			max_samples=uniform(.1,.9)
		)
	elif model_type=="lgbm_class":
		model = lgb.LGBMClassifier(random_state=0, num_threads=1)
		distributions = dict(
			n_estimators = randint(10,1000),
			learning_rate = loguniform(a=.001,b=1),
			subsample = uniform(.1,.9),
			colsample_bytree = uniform(.1,.9),
			reg_lambda = loguniform(a=.001,b=100),
			num_leaves = randint(8,64)
		)

	elif model_type=="lgbm_reg":
		model = lgb.LGBMRegressor(random_state=0, num_threads=1)
		distributions = dict(
			n_estimators = randint(10,1000),
			learning_rate = loguniform(a=.001,b=1),
			subsample = uniform(.1,.9),
			colsample_bytree = uniform(.1,.9),
			reg_lambda = loguniform(a=.001,b=100),
			num_leaves = randint(8,64)
		)

	clf = RandomizedSearchCV(model, distributions, random_state=0, scoring=scoring, n_iter=100, n_jobs=8 if not model in ["lgbm_reg","lgbm_class"] else 1)
	search = clf.fit(X_train_temp,y_train)
	logging.info(f"Best hyperparameters found: {search.best_params_}")
	logging.info(f"CV score of best hyperparameters: {search.best_score_}")

	return search.best_params_, search.best_score_


