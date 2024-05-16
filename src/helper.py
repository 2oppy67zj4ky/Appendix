from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
import numpy as np
import json
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold
import time
import ray
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from knncmi import *
from sklearnex import patch_sklearn

import pickle

from sklearn.utils.validation import (check_array, check_consistent_length,
                                _num_samples)
from sklearn.utils.validation import column_or_1d
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import OneHotEncoder

#k-nearest neighbors approach to calculating interaction information
@ray.remote
def int_gain_knn(data,cola,colb,coly,k):
	int_inf = cmi(cola,colb,coly,k,data) - cmi(cola,colb,[],k,data)
	return int_inf


def train_test(X,train_size,shuffle):
	#for computational efficiency, we can use a subset of the training data
	X_train = X[0:train_size,:].copy()
	X_train = X_train[shuffle]

	X_test = X[train_size:,:].copy()

	return X_train, X_test, shuffle

def add_features(combo_type, op, X, el1, el2):
	is_added=True
	X_train,X_test = X
	if combo_type == "numnum":
		if op=="mul":
			newcol_train = np.multiply(X_train[:,el1].copy(), X_train[:,el2].copy())
			newcol_test = np.multiply(X_test[:,el1].copy(), X_test[:,el2].copy())
		elif op == "add":
			newcol_train = X_train[:,el1].copy() + X_train[:,el2].copy()
			newcol_test = X_test[:,el1].copy() + X_test[:,el2].copy()
		elif op == "sub1":
			newcol_train = X_train[:,el1].copy() - X_train[:,el2].copy()
			newcol_test = X_test[:,el1].copy() - X_test[:,el2].copy()
		elif op == "sub2":
			newcol_train = X_train[:,el2].copy() - X_train[:,el1].copy()
			newcol_test = X_test[:,el2].copy() - X_test[:,el1].copy()
		elif op == "min":
			newcol_train = np.min(np.hstack((X_train[:,el2].copy().reshape((-1,1)),X_train[:,el1].copy().reshape((-1,1)))), axis=1)
			newcol_test = np.min(np.hstack((X_test[:,el2].copy().reshape((-1,1)),X_test[:,el1].copy().reshape((-1,1)))), axis=1)
		elif op == "max":
			newcol_train = np.max(np.hstack((X_train[:,el2].copy().reshape((-1,1)),X_train[:,el1].copy().reshape((-1,1)))), axis=1)
			newcol_test = np.max(np.hstack((X_test[:,el2].copy().reshape((-1,1)),X_test[:,el1].copy().reshape((-1,1)))), axis=1)
		elif op == "div1":
			if (np.sum(np.isclose(X_train[:,el2], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el2], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = np.divide(X_train[:,el1].copy(), X_train[:,el2].copy())
				newcol_test = np.divide(X_test[:,el1].copy(), X_test[:,el2].copy())
			else:
				is_added=False
		elif op == "div1absplus1":
			newcol_train = np.divide(X_train[:,el1].copy(), np.abs(X_train[:,el2].copy())+1)
			newcol_test = np.divide(X_test[:,el1].copy(), np.abs(X_test[:,el2].copy())+1)
		elif op == "div2absplus1":
			newcol_train = np.divide(X_train[:,el2].copy(), np.abs(X_train[:,el1].copy())+1)
			newcol_test = np.divide(X_test[:,el2].copy(), np.abs(X_test[:,el1].copy())+1)

		elif op == "div2":
			if (np.sum(np.isclose(X_train[:,el1], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el1], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = np.divide(X_train[:,el2].copy(), X_train[:,el1].copy())
				newcol_test = np.divide(X_test[:,el2].copy(), X_test[:,el1].copy())
			else:
				is_added=False
		elif op == "mod1":
			if (np.sum(np.isclose(X_train[:,el2], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el2], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = X_train[:,el1].copy() % X_train[:,el2].copy()
				newcol_test = X_test[:,el1].copy() % X_test[:,el2].copy()
			else:
				is_added=False
		elif op == "mod2":
			if (np.sum(np.isclose(X_train[:,el1], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el1], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = X_train[:,el2].copy() % X_train[:,el1].copy()
				newcol_test = X_test[:,el2].copy() % X_test[:,el1].copy()
			else:
				is_added=False
		vartype="num"
	elif combo_type=="numcat" or combo_type=="catnum":
		if combo_type=="catnum":
			el2_c=el2
			el2=el1
			el1=el2_c
		if op=="group_then_min":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				min_train = np.min(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = min_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = min_train
				
		elif op=="group_then_max":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				max_train = np.max(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = max_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = max_train
		elif op=="group_then_mean":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				mean_train = np.mean(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = mean_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = mean_train
		elif op=="group_then_median":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				median_train = np.median(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = median_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = median_train
		elif op=="group_then_std":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				std_train = np.std(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = std_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = std_train
		elif op=="mul":
			newcol_train = np.multiply(X_train[:,el1].copy(), X_train[:,el2].copy())
			newcol_test = np.multiply(X_test[:,el1].copy(), X_test[:,el2].copy())
		elif op == "add":
			newcol_train = X_train[:,el1].copy() + X_train[:,el2].copy()
			newcol_test = X_test[:,el1].copy() + X_test[:,el2].copy()
		elif op == "sub1":
			newcol_train = X_train[:,el1].copy() - X_train[:,el2].copy()
			newcol_test = X_test[:,el1].copy() - X_test[:,el2].copy()
		elif op == "sub2":
			newcol_train = X_train[:,el2].copy() - X_train[:,el1].copy()
			newcol_test = X_test[:,el2].copy() - X_test[:,el1].copy()
		elif op == "min":
			newcol_train = np.min(np.hstack((X_train[:,el2].copy().reshape((-1,1)),X_train[:,el1].copy().reshape((-1,1)))), axis=1)
			newcol_test = np.min(np.hstack((X_test[:,el2].copy().reshape((-1,1)),X_test[:,el1].copy().reshape((-1,1)))), axis=1)
		elif op == "max":
			newcol_train = np.max(np.hstack((X_train[:,el2].copy().reshape((-1,1)),X_train[:,el1].copy().reshape((-1,1)))), axis=1)
			newcol_test = np.max(np.hstack((X_test[:,el2].copy().reshape((-1,1)),X_test[:,el1].copy().reshape((-1,1)))), axis=1)
		elif op == "div1":
			if (np.sum(np.isclose(X_train[:,el2], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el2], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = np.divide(X_train[:,el1].copy(), X_train[:,el2].copy())
				newcol_test = np.divide(X_test[:,el1].copy(), X_test[:,el2].copy())
			else:
				is_added=False
		elif op == "div1absplus1":
			newcol_train = np.divide(X_train[:,el1].copy(), np.abs(X_train[:,el2].copy())+1)
			newcol_test = np.divide(X_test[:,el1].copy(), np.abs(X_test[:,el2].copy())+1)
		elif op == "div2absplus1":
			newcol_train = np.divide(X_train[:,el2].copy(), np.abs(X_train[:,el1].copy())+1)
			newcol_test = np.divide(X_test[:,el2].copy(), np.abs(X_test[:,el1].copy())+1)
		elif op == "div2":
			if (np.sum(np.isclose(X_train[:,el1], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el1], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = np.divide(X_train[:,el2].copy(), X_train[:,el1].copy())
				newcol_test = np.divide(X_test[:,el2].copy(), X_test[:,el1].copy())
			else:
				is_added=False
		elif op == "mod1":
			if (np.sum(np.isclose(X_train[:,el2], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el2], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = X_train[:,el1].copy() % X_train[:,el2].copy()
				newcol_test = X_test[:,el1].copy() % X_test[:,el2].copy()
			else:
				is_added=False
		elif op == "mod2":
			if (np.sum(np.isclose(X_train[:,el1], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el1], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = X_train[:,el2].copy() % X_train[:,el1].copy()
				newcol_test = X_test[:,el2].copy() % X_test[:,el1].copy()
			else:
				is_added=False
		#switch back
		if combo_type=="catnum":
			el2_c=el2
			el2=el1
			el1=el2_c
		vartype="num"

	elif combo_type=="catcat":
		vartype="num"
		if op=="group_then_min":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				min_train = np.min(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = min_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = min_train
				
		elif op=="group_then_max":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				max_train = np.max(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = max_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = max_train
		elif op=="group_then_mean":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				mean_train = np.mean(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = mean_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = mean_train
		elif op=="group_then_median":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				median_train = np.median(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = median_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = median_train
		elif op=="group_then_std":
			unique_vals_train = np.unique(X_train[:,el2])
			newcol_train = np.zeros(X_train.shape[0])
			newcol_test = np.zeros(X_test.shape[0])
			for u in unique_vals_train:
				col_train = X_train[:,el2]
				iso_train=col_train
				col_train =X_train[np.where(col_train==u)[0],el1]
				std_train = np.std(col_train)
				col_test = X_test[:,el2]
				iso_test=col_test
				newcol_test[np.where(iso_test==u)[0]] = std_train
				col_train = X_train[:,el2]
				iso_train=col_train
				newcol_train[np.where(iso_test==u)[0]] = std_train
		if op=="mul":
			newcol_train = np.multiply(X_train[:,el1].copy(), X_train[:,el2].copy())
			newcol_test = np.multiply(X_test[:,el1].copy(), X_test[:,el2].copy())
		elif op == "add":
			newcol_train = X_train[:,el1].copy() + X_train[:,el2].copy()
			newcol_test = X_test[:,el1].copy() + X_test[:,el2].copy()
		elif op == "sub1":
			newcol_train = X_train[:,el1].copy() - X_train[:,el2].copy()
			newcol_test = X_test[:,el1].copy() - X_test[:,el2].copy()
		elif op == "sub2":
			newcol_train = X_train[:,el2].copy() - X_train[:,el1].copy()
			newcol_test = X_test[:,el2].copy() - X_test[:,el1].copy()
		elif op == "min":
			newcol_train = np.min(np.hstack((X_train[:,el2].copy().reshape((-1,1)),X_train[:,el1].copy().reshape((-1,1)))), axis=1)
			newcol_test = np.min(np.hstack((X_test[:,el2].copy().reshape((-1,1)),X_test[:,el1].copy().reshape((-1,1)))), axis=1)
		elif op == "max":
			newcol_train = np.max(np.hstack((X_train[:,el2].copy().reshape((-1,1)),X_train[:,el1].copy().reshape((-1,1)))), axis=1)
			newcol_test = np.max(np.hstack((X_test[:,el2].copy().reshape((-1,1)),X_test[:,el1].copy().reshape((-1,1)))), axis=1)
		elif op == "div1":
			if (np.sum(np.isclose(X_train[:,el2], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el2], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = np.divide(X_train[:,el1].copy(), X_train[:,el2].copy())
				newcol_test = np.divide(X_test[:,el1].copy(), X_test[:,el2].copy())
			else:
				is_added=False
		elif op == "div1absplus1":
			newcol_train = np.divide(X_train[:,el1].copy(), np.abs(X_train[:,el2].copy())+1)
			newcol_test = np.divide(X_test[:,el1].copy(), np.abs(X_test[:,el2].copy())+1)
		elif op == "div2absplus1":
			newcol_train = np.divide(X_train[:,el2].copy(), np.abs(X_train[:,el1].copy())+1)
			newcol_test = np.divide(X_test[:,el2].copy(), np.abs(X_test[:,el1].copy())+1)
		elif op == "div2":
			if (np.sum(np.isclose(X_train[:,el1], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el1], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = np.divide(X_train[:,el2].copy(), X_train[:,el1].copy())
				newcol_test = np.divide(X_test[:,el2].copy(), X_test[:,el1].copy())
			else:
				is_added=False
		elif op == "mod1":
			if (np.sum(np.isclose(X_train[:,el2], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el2], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = X_train[:,el1].copy() % X_train[:,el2].copy()
				newcol_test = X_test[:,el1].copy() % X_test[:,el2].copy()
			else:
				is_added=False
		elif op == "mod2":
			if (np.sum(np.isclose(X_train[:,el1], np.zeros(X_train.shape[0]))) == 0
				and np.sum(np.isclose(X_test[:,el1], np.zeros(X_test.shape[0]))) == 0
				):
				newcol_train = X_train[:,el2].copy() % X_train[:,el1].copy()
				newcol_test = X_test[:,el2].copy() % X_test[:,el1].copy()
			else:
				is_added=False
		elif op=="combine_then_freq":
			is_added=False
		elif op=="group_then_n_unique1":
			is_added=False
		elif op=="group_then_n_unique2":
			is_added=False
		else:
			is_added=False
	else:
		#for now only do num-num transformations
		is_added=False


	if is_added:
		Xtemp_train = np.append(X_train.copy(),newcol_train.reshape(X_train.shape[0],1),axis=1)
		Xtemp_test = np.append(X_test.copy(),newcol_test.reshape(X_test.shape[0],1),axis=1)
	else:
		Xtemp_train = X_train.copy()
		Xtemp_test = X_test.copy()
	
	if is_added:
		trans_string = str(el1) + " " + str(el2) + ":" + combo_type + ":" + op
	else:
		trans_string = "NONE"
		vartype=None

	return Xtemp_train, Xtemp_test, trans_string, [el1,el2], vartype

@ray.remote
def eval_features(clf,X_train, y_train, scoring, feats,model,vartype_list,extra_type, seed=0):
	#train and evaluate performance
	#clf = RandomForestClassifier(n_estimators = 10, random_state=0)
	# scaler = StandardScaler()
	# scaler.fit(X_train_temp.copy())
	# X_train_temp=scaler.transform(X_train_temp.copy())
	if model == "LR" or model =="lasso":
		valscore = val_score(clf,scoring,X_train,y_train,vartype_list,normalize=True, ohe=True, extra_type=extra_type, seed=seed)
	else:
		valscore = val_score(clf,scoring,X_train,y_train,vartype_list, extra_type=extra_type, seed=seed)
	return valscore, feats

# change the operations you wish to include in the algorithm here
def ops(combo_type):
	if combo_type == "numnum":
		# return ["add","mul","div1","mod1","mod2","sub1","min","max","div1absplus1","div2absplus1"]
		return ["add","mul","sub1","min","max","div1absplus1","div2absplus1"]
	elif combo_type == "numcat" or combo_type=="catnum":
		# return ["add","mul","div1","mod1","mod2","sub1","min","max","div1absplus1","div2absplus1"]
		# return ["group_then_min","group_then_max","group_then_mean","group_then_median","group_then_std","add","mul","div1","mod1","mod2","sub1","min","max","div1absplus1","div2absplus1"]
		return ["group_then_min","group_then_max","group_then_mean","group_then_median","group_then_std","add","mul","sub1","min","max","div1absplus1","div2absplus1"]
	else:
		# return ["group_then_min","group_then_max","group_then_mean","group_then_median","group_then_std","add","mul","div1","mod1","mod2","sub1","min","max","div1absplus1","div2absplus1"]
		return ["group_then_min","group_then_max","group_then_mean","group_then_median","group_then_std","add","mul","sub1","min","max","div1absplus1","div2absplus1"]


def RAE_comp(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (np.abs(y_true - y_pred)).sum(axis=0, dtype=np.float64)
    denominator = (np.abs(y_true - np.average(y_true))).sum(axis=0, dtype=np.float64)

    # print(np.average(y_true))
    # print(numerator)
    # print(denominator)
    # print(1 - numerator / denominator)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)


def RAE_comp_w_est(estimator, X22, y_true):
    sample_weight=None
    multioutput="uniform_average"
    y_pred = estimator.predict(X22)
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (np.abs(y_true - y_pred)).sum(axis=0, dtype=np.float64)
    denominator = (np.abs(y_true - np.average(y_true))).sum(axis=0, dtype=np.float64)

    # print(np.average(y_true))
    # print(numerator)
    # print(denominator)
    # print(1 - numerator / denominator)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)

def _check_reg_targets(y_true, y_pred, multioutput):
    """Check that y_true and y_pred belong to the same regression task

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'

    y_true : array-like of shape = (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape = (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.

    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


def val_score(clf,scoring,X_train,y_train,vartype_list, normalize=False, ohe=False, extra_type=None, seed=0):
    if extra_type:
        vartype_list.append(extra_type)
    if ohe:
        if "cat" in vartype_list:
            enc = OneHotEncoder(handle_unknown='ignore')
            X_train_num = X_train[:,np.where(np.array(vartype_list)=="num")[0]]
            X_train_cat = X_train[:,np.where(np.array(vartype_list)=="cat")[0]]
            enc.fit(X_train_cat)
            X_train_ohe = enc.transform(X_train_cat).toarray()
            X_train = np.concatenate((X_train_num,X_train_ohe),axis=1)
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train=scaler.transform(X_train)

    if scoring=="RAE_comp":
        scoring=make_scorer(RAE_comp)

    kfld = KFold(n_splits=5,shuffle=True, random_state=seed)
    valscore = np.mean(cross_val_score(clf,X_train,y_train,scoring=scoring,cv=kfld))
    
    return valscore



def unary(clf, X_train, y_train, X_test, scoring, vartype_list, spot=None, seed=0):
	#now we test unary transformations of the new feature
    #TODO: Need to make this into a function form to have less repeat code...
    if spot:
    	m=spot
    else:
    	m = X_train.shape[1]-1
    list_X_train_unary = []
    list_X_test_unary = []
    #store list of scores on validation
    list_scores_unary = []
    list_type_unary=[]
    #now add unary transformations
    #try each unary transformation on each new feature and keep the one with best cv score
    X_train_temp = X_train.copy()
    X_test_temp = X_test.copy()
    #train and evaluate performance
    #clf = RandomForestClassifier(n_estimators = 10, random_state=0)
    normalize=True
    ohe=True
    valscore = val_score(clf,scoring,X_train_temp,y_train,vartype_list,normalize=normalize, ohe=ohe, seed=seed)
    list_scores_unary.append(valscore)
    list_X_train_unary.append(X_train_temp.copy())
    list_X_test_unary.append(X_test_temp.copy())
    list_type_unary.append("None")
    

    X_train_temp = X_train.copy()
    X_test_temp = X_test.copy()
    X_train_temp[:,m] = np.multiply(X_train_temp[:,m], X_train_temp[:,m])
    X_test_temp[:,m] = np.multiply(X_test_temp[:,m], X_test_temp[:,m])
    #train and evaluate performance
    #clf = RandomForestClassifier(n_estimators = 10, random_state=0)

    valscore = val_score(clf,scoring,X_train_temp,y_train,vartype_list,normalize=normalize, ohe=ohe, seed=seed)
    list_scores_unary.append(valscore)
    list_X_train_unary.append(X_train_temp.copy())
    list_X_test_unary.append(X_test_temp.copy())
    list_type_unary.append("Square")

    # if (np.sum(np.isclose(X_train[:,m], np.zeros(X_train.shape[0]))) == 0
    #     and np.sum(np.isclose(X_test[:,m], np.zeros(X_test.shape[0]))) == 0
    #     ):
    #     X_train_temp = X_train.copy()
    #     X_test_temp = X_test.copy()
    #     X_train_temp[:,m] = 1./X_train_temp[:,m]
    #     X_test_temp[:,m] = 1./X_test_temp[:,m]
        
    #     #train and evaluate performance
    #     valscore = val_score(clf,scoring,X_train_temp,y_train,vartype_list,normalize=normalize, ohe=ohe, seed=seed)
    #     list_scores_unary.append(valscore)
    #     list_X_train_unary.append(X_train_temp.copy())
    #     list_X_test_unary.append(X_test_temp.copy())
    #     list_type_unary.append("Reciprocal")


    X_train_temp = X_train.copy()
    X_test_temp = X_test.copy()
    X_train_temp[:,m] = np.abs(X_train_temp[:,m])
    X_test_temp[:,m] = np.abs(X_test_temp[:,m])
    #train and evaluate performance
    #clf = RandomForestClassifier(n_estimators = 10, random_state=0)

    valscore = val_score(clf,scoring,X_train_temp,y_train,vartype_list,normalize=normalize, ohe=ohe, seed=seed)
    list_scores_unary.append(valscore)
    list_X_train_unary.append(X_train_temp.copy())
    list_X_test_unary.append(X_test_temp.copy())
    list_type_unary.append("Abs")  


    X_train_temp = X_train.copy()
    X_test_temp = X_test.copy()
    X_train_temp[:,m] = np.sqrt(np.abs(X_train_temp[:,m]))
    X_test_temp[:,m] = np.sqrt(np.abs(X_test_temp[:,m]))
    #train and evaluate performance
    #clf = RandomForestClassifier(n_estimators = 10, random_state=0)

    valscore = val_score(clf,scoring,X_train_temp,y_train,vartype_list,normalize=normalize, ohe=ohe, seed=seed)
    list_scores_unary.append(valscore)
    list_X_train_unary.append(X_train_temp.copy())
    list_X_test_unary.append(X_test_temp.copy())
    list_type_unary.append("Sqrt Abs") 


    X_train_temp = X_train.copy()
    X_test_temp = X_test.copy()
    X_train_temp[:,m] = 1/(1+np.exp(-X_train_temp[:,m]))
    X_test_temp[:,m] = 1/(1+np.exp(-X_test_temp[:,m]))
    #train and evaluate performance
    #clf = RandomForestClassifier(n_estimators = 10, random_state=0)

    valscore = val_score(clf,scoring,X_train_temp,y_train,vartype_list,normalize=normalize, ohe=ohe, seed=seed)
    list_scores_unary.append(valscore)
    list_X_train_unary.append(X_train_temp.copy())
    list_X_test_unary.append(X_test_temp.copy())
    list_type_unary.append("sigmoid")        

    #find the best addition
    top_valscore_unary = np.argmax(np.array(list_scores_unary))

    X_train = list_X_train_unary[top_valscore_unary]
    X_test = list_X_test_unary[top_valscore_unary]
    return X_train, X_test


