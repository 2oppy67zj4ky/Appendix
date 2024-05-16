def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import sys
file_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, str(file_path))
from iife import iife, blockPrint, enablePrint
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
from sklearn.metrics import f1_score, mean_squared_error, r2_score, make_scorer, mean_absolute_error, log_loss
from sklearn.model_selection import KFold
import time
import ray
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from knncmi import *
import pickle
from sklearn.preprocessing import OneHotEncoder
from hyperparam_tune import hyperparam_tune
import logging
import lightgbm as lgb
from time import sleep
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
runs = [
("openml586","lasso")
]
output_dir = "outdir/iife_outputs/"
save_Xs=True
loss_before=[]
loss_after=[]
scores_before=[]
scores_after=[]
hyperparams_params=[]
hyperparams_params_after=[]
hyperparams_scores=[]
times=[]
#set some ray settings to work properly
runtime_env = {"working_dir": "src"}
ray.init(runtime_env=runtime_env, num_cpus=6)
# run on 4 total runs for this test run
num_seeds=2
num_seed2s=2
for it,r in enumerate(runs):
	for seed in range(0,num_seeds):
		task,model=r
		#####
		##### load in data
		#####
		if task == "openml586":
			X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_openml586(seed)
		elif task == "cal_housing":
			X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_cal_housing(seed)
		elif task == "jungle_chess":
			X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_jungle_chess(seed)
		train_size = X_train.shape[0]
		y_train = np.array(y_train)
		y_test = np.array(y_test)

		#####
		##### Normalize and one-hot encode data if linear model
		#####
		if model == "LR" or model == "lasso":
			if "cat" in vartype_list:
				# one hot encode
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
		
		#####
		##### Hyperparameter tune before autofe
		#####
		print("Starting hyperparam tuning")
		hyperparams,hyperscores = hyperparam_tune(task,model,seed)
		print("Ending hyperparam tuning")
		hyperparams_params.append(hyperparams)
		hyperparams_scores.append(hyperscores)

		if model == "LR":
			clf = LogisticRegression(C=hyperparams["C"],max_iter=50000)
		elif model == "RFR":
			clf = RandomForestRegressor(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
		elif model == "RF":
			clf = RandomForestClassifier(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
		elif model == "lasso":
			clf = Lasso(alpha=hyperparams["alpha"],max_iter=50000)
		elif model == "lgbm_reg":
			clf = lgb.LGBMRegressor(
				n_estimators = hyperparams["n_estimators"],
				learning_rate = hyperparams["learning_rate"],
				subsample = hyperparams["subsample"],
				colsample_bytree = hyperparams["colsample_bytree"],
				reg_lambda = hyperparams["reg_lambda"], 
				random_state=0, num_threads=1
			)
		elif model == "lgbm_class":
			clf = lgb.LGBMClassifier(
				n_estimators = hyperparams["n_estimators"],
				learning_rate = hyperparams["learning_rate"],
				subsample = hyperparams["subsample"],
				colsample_bytree = hyperparams["colsample_bytree"],
				reg_lambda = hyperparams["reg_lambda"], 
				random_state=0, num_threads=1
			)

		#####
		##### Find test scores before AutoFE
		#####

		clf.fit(X_train_temp, y_train)
		pred = clf.predict(X_test_temp)

		if model in ["lasso","RFR","lgbm_reg"]:
			score = RAE_comp(y_test,pred)
		else:
			score = f1_score(y_test,pred,average="micro")


		scores_before.append(score)

		if model=="LR":
			loss_before.append(log_loss(y_train,clf.predict_proba(X_train_temp)) )

		for seed2 in range(0,num_seed2s):
			if task == "openml586":
				X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_openml586(seed)
			elif task == "cal_housing":
				X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_cal_housing(seed)
			elif task == "jungle_chess":
				X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_jungle_chess(seed)
			train_size = X_train.shape[0]
			y_train = np.array(y_train)
			y_test = np.array(y_test)

			data_input = (X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list)
			# blockPrint()
			start1=time.time()
			if model=="lasso":
				clf.set_params(max_iter=100)
			elif model=="LR":
				clf.set_params(max_iter=100)
			if model in ["lasso","RFR","lgbm_reg"]:
				scoring="RAE_comp"
			else:
				scoring='f1_micro'

			#####
			##### The AutoFE process is performed below with the iife() function
			#####	

			X_train, y_train, X_test, vartype_list, cvs, operation_list = iife(data_input = data_input, model = model, clf=clf, scoring=scoring, K=3, task = task, patience=20 if not task in ["cal_housing","fri"] else 40, int_inf_subset = 3000, eps=0, simul=False, seed=seed, seed2=seed2)

			#####
			##### Write intermediate outputs to file
			#####

			f = open(output_dir + f"iife_validation_per_iter_{task}_{model}_{seed}_{seed2}.txt", "w")
			f.write(str(cvs))
			f.close()

			f = open(output_dir + f"iife_operation_list_{task}_{model}_{seed}_{seed2}.txt", "w")
			f.write(str(operation_list))
			f.close()

			#####
			##### Hyperparameter tune AGAIN after AutoFE
			#####
			data=(X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list)
			hyperparams, hyperscores = hyperparam_tune(task,model,seed, data=data)
			hyperparams_params_after.append(hyperparams)

			if model == "LR":
				clf = LogisticRegression(C=hyperparams["C"],max_iter=50000)
			elif model == "RFR":
				clf = RandomForestRegressor(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
			elif model == "RF":
				clf = RandomForestClassifier(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
			elif model == "lasso":
				clf = Lasso(alpha=hyperparams["alpha"],max_iter=500000)
			elif model == "lgbm_reg":
				clf = lgb.LGBMRegressor(
					n_estimators = hyperparams["n_estimators"],
					learning_rate = hyperparams["learning_rate"],
					subsample = hyperparams["subsample"],
					colsample_bytree = hyperparams["colsample_bytree"],
					reg_lambda = hyperparams["reg_lambda"], 
					random_state=0, num_threads=1
				)
			elif model == "lgbm_class":
				clf = lgb.LGBMClassifier(
					n_estimators = hyperparams["n_estimators"],
					learning_rate = hyperparams["learning_rate"],
					subsample = hyperparams["subsample"],
					colsample_bytree = hyperparams["colsample_bytree"],
					reg_lambda = hyperparams["reg_lambda"], 
					random_state=0, num_threads=1
				)


			if model == "LR" or model == "lasso":
				if "cat" in vartype_list:
					# one hot encode
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

			#####
			##### Find test scores after AutoFE
			#####

			clf.fit(X_train_temp, y_train)
			pred = clf.predict(X_test_temp)

			if model=="LR":
				loss_after.append(log_loss(y_train,clf.predict_proba(X_train_temp)) ) #+ np.linalg.norm(clf.coef_,2)

			if model in ["lasso","RFR","lgbm_reg"]:
				score = RAE_comp(y_test,pred)
			else:
				score = f1_score(y_test,pred,average="micro")
			scores_after.append(score)
			end1=time.time()
			times.append(end1-start1)


			logging.info(f"Score before AutoFE: {scores_before[-1]}")
			logging.info(f"Score before AutoFE: {scores_after[-1]}")
			logging.info(f"AutoFE Time spent: {times[-1]}")

			#####
			##### Store results of this runs
			#####
			f = open(output_dir + f"iife_{task}_{model}_{seed}_{seed2}.txt", "w")
			f.write("Baseline test score: " + str(scores_before[-1]))
			f.write("\n Transformed test score: " + str(scores_after[-1]))
			f.write("\n Time: " + str(times[-1]))
			f.close()


			f = open(output_dir + f"iife_vartype_list_{task}_{model}_{seed}_{seed2}.txt", "w")
			f.write(str(vartype_list))
			f.close()

			with open(output_dir + f'iife_X_train_{task}_{model}_{seed}_{seed2}.npy', 'wb') as f:
				np.save(f, X_train)


			with open(output_dir + f'iife_X_test_{task}_{model}_{seed}_{seed2}.npy', 'wb') as f:
				np.save(f, X_test)

	logging.info(f"Scores before AutoFE: {scores_before}")
	logging.info(f"Scores before AutoFE: {scores_after}")
	logging.info(f"AutoFE Time spent: {times}")

	#####
	##### Store the results of the entire run
	#####
	f = open(output_dir + f"iife_{task}_{model}_FULL_run.txt", "w")
	f.write("Baseline test scores: " + str(scores_before))
	f.write("\n Transformed test scores: " + str(scores_after))
	f.write("\n Times: " + str(times))
	f.close()
	scores_before=[]
	scores_after=[]
	times=[]




