### You can place your own data loading functions here
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

#imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
import numpy as np
import json
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold
import time
import ray
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from knncmi import cmi
from sklearnex import patch_sklearn
import pickle


def process_openml586(seed):
	import arff
	dataset = arff.load(open('data/openml586.arff', 'r'))
	data = np.array(dataset['data'])
	y=data[:,-1]
	X=data[:,:-1].astype('float')
	X_orig = X.copy()
	#now handle the one-hot encodings from the json specifications
	vartype_list=["num"]*X.shape[1]
	for i in range(0,X.shape[1]):
		if np.unique(X[:,i]).shape[0]<=250:
			vartype_list[i]="ord"
	feat_list=None
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed)
	# for i in range(0,len(vartype_list)):
	# 	vartype_list[i]="num"
	return X_train,y_train, X_test,y_test,feat_list,X_orig,vartype_list

def process_cal_housing(seed):
	import sklearn
	caldata = sklearn.datasets.fetch_california_housing()
	X=caldata.data
	y=caldata.target
	X_orig = X.copy()
	y_orig=y.copy()
	#now handle the one-hot encodings from the json specifications
	vartype_list=["num"]*X.shape[1]
	for i in range(0,X.shape[1]):
		if np.unique(X[:,i]).shape[0]<=250:
			vartype_list[i]="ord"
	feat_list=None
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed)
	# for i in range(0,len(vartype_list)):
	# 	vartype_list[i]="num"
	return X_train,y_train, X_test,y_test,feat_list,X_orig,vartype_list


def process_jungle_chess(seed):
	import arff
	dataset = arff.load(open('data/jungle_chess.arff', 'r'))
	data = np.array(dataset['data'])
	y=data[:,-1]
	X=data[:,:-1].astype('float')
	X=np.nan_to_num(X)
	X_orig = X.copy()
	#now handle the one-hot encodings from the json specifications
	vartype_list=["num"]*X.shape[1]
	for i in range(0,X.shape[1]):
		if np.unique(X[:,i]).shape[0]<=250:
			vartype_list[i]="ord"
	feat_list=None
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed)
	# for i in range(0,len(vartype_list)):
	# 	vartype_list[i]="num"
	return X_train,y_train, X_test,y_test,feat_list,X_orig,vartype_list

# function below one hot encodes the categorical variables specified in the json file and returns the final data matrix
def one_hot_enc(X,url_json):
	X_old = X.copy()
	with open(url_json) as j:
		var_types = json.load(j)
	index_on=0
	feat_list=[]
	vartype_list = []
	for i,v in enumerate(var_types):
		vartype_list.append(var_types[v])
		# one-hot encode if categorical
		if var_types[v]=="cat":
			#find number of categories
			
			catsunique = np.unique(X_old[:,i])
			numcats = catsunique.shape[0]
			#now add in the one-hot encoded vector in place of the numerical value
			#first add in empty zeros
			X = np.delete(X,index_on,axis=1)
			n=list((np.ones(numcats)*index_on).astype('int'))
			X = np.insert(X,n,0,axis=1)
			# now add 1 for the encodings
			for z in range(0,X.shape[0]):
				cat = X_old[z,i]
				X[z,int(index_on+np.where(catsunique==cat)[0])]=1
			feat_list.append(list(range(index_on, index_on+numcats)))
			index_on += numcats
		else:
			feat_list.append(index_on)
			index_on+=1

	return X, feat_list, vartype_list



