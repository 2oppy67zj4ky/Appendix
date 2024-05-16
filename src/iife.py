from data import *
from helper import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
import numpy as np
import json
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import time
import ray
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from knncmi import *
import pickle
import os
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
import sys, os
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

###################################################################
### Main call for IIFE
### INPUTS:
# model: str that specifies which downstream model to use. options supported are "lasso", "RF" (RF classification), "RFR" (regression), "LR" (logistic regression)
# task: which task we are performing autofe on. Typically specifies the data to use and the scoring metric to use. The function calls for the data processing is in data.py
# K: number of feature pairs to consider each iteration (usually kept low for very large datasets)
# patience: number of iterations to allow for the validation score to decrease before terminating 
# n_int_inf: number of features to consider when calculating interaction information (scales like n choose 2 so be careful when selecting this)
# int_inf_subset: subset of samples used for calculating interaction information
# num_runs: number of times we do the whole AutoFE process
###################################################################

def iife(data_input, model, clf, task, scoring="f1_micro", K=1, patience=3, n_int_inf=500, int_inf_subset = 5000, eps=0, num_runs=1, simul=False, seed=0, seed2=0):
    cvs =[]
    testf1s = []
    basecvf1s = []
    basetestf1s = []
    timings = []
    
    n_est= 10
    writetofile = True
    num_top = n_int_inf
    maxiterlr=2000
    tol=1e-2
    
    average= "micro"
    n_jobs=1
    solver='lbfgs'
    alpha=1
    l1_ratio=0.5
    knnk=4
    iisubsetsize=int_inf_subset
    RF_coeff = True
    lincoeffs=[]
    feat_length = []
    operation_list=[]
    unary_operation_list=[]
    

    bigstart1=time.time()

    #set seed to run number
    np.random.seed(seed)
    
    X_train,y_train,X_test,y_test,feat_list,X_orig,vartype_list = data_input
    original_feature_size = X_train.shape[1]
    

    #we may decide to use a subset of the training size for each run
    train_size=X_train.shape[0]
    subset_size=train_size
    feat_length.append(X_train.shape[1])
    M=X_train.shape[1]
    logging.info(f"X_train shape: {X_train.shape}")
    X_orig_train = X_train.copy()
    X_orig_test = X_test.copy()

    # split train into validation


    #max iters
    iters=X_train.shape[1]*100
    val_accs=[]
    val_accs_test=[]
    original_X_train = X_train.copy()


    cv = val_score(clf,scoring,X_train,y_train,vartype_list,seed=seed2*100)
    lastvalscore = cv
    cvs.append(cv)

    pc=0
    explored_pairs=[]

    #store the interaction information so we do not recompute it every time
    interaction_information = []
    interaction_pairs = []

    #calculate the interaction information in the original features (or subset of them)
    logging.info("Calculating interaction information")
    start1 = time.time()


    xdf1=np.concatenate((X_train,np.reshape(y_train, (y_train.shape[0],1))),axis=1)
    xdf = pd.DataFrame(data=xdf1.copy())
    xdf = xdf.sample(n=np.min([X_train.shape[0],iisubsetsize]), random_state=seed2).reset_index(drop=True)
    feat_pairs = [[f1,f2]  for f1,f2 in combinations(range(0,M), 2)]
    futures = [int_gain_knn.remote(xdf.iloc[:,[f1,f2,xdf1.shape[1]-1]],[0],[1],[2],knnk) for f1,f2 in combinations(range(0,M), 2)]

    int_vals = ray.get(futures)

    interaction_information.extend(int_vals)
    interaction_pairs.extend(feat_pairs)

    end1 = time.time()
    logging.info("Feature interaction information time spent: " + str(end1-start1))
    n_lags=patience
    stored_cv=[cv]
    is_feat_added=False
    for it in range(0,iters):
        #create two train, val splits

        #now only compute the interaction information between features that are new
        xdf1=np.concatenate((X_train,np.reshape(y_train, (y_train.shape[0],1))),axis=1)
        xdf = pd.DataFrame(data=xdf1.copy())
        xdf = xdf.sample(n=np.min([X_train.shape[0],iisubsetsize]), random_state=seed2).reset_index(drop=True)
        if it>0 and is_feat_added:
            if num_top <= X_train.shape[1]:
                futures = [int_gain_knn.remote(xdf.iloc[:,[f1,X_train.shape[1]-1,xdf1.shape[1]-1]],[0],[1],[2],k=knnk) for f1 in range(0,X_train.shape[1]-1)]
                feat_pairs = [[f1,X_train.shape[1]-1]  for f1 in range(0,X_train.shape[1]-1)]
            else:
                futures = [int_gain_knn.remote(xdf.iloc[:,[f1,X_train.shape[1]-1,xdf1.shape[1]-1]],[0],[1],[2],k=knnk) for f1 in range(0,X_train.shape[1]-1)]
                feat_pairs = [[f1,X_train.shape[1]-1]  for f1 in range(0,X_train.shape[1]-1)]
            int_vals = ray.get(futures)

            interaction_information.extend(int_vals)
            interaction_pairs.extend(feat_pairs)

        #now find the top scoring interaction informations
        sorteds = np.argsort(-np.array(interaction_information))
        
        start2 = time.time()

        #store list of Xs
        list_X_train = []
        list_X_test = []
        #store list of scores on validation
        list_scores = []
        list_trans_strings=[]
        list_feat_pairs=[]
        list_feats=[]
        list_types=[]
        #find the new features using a ray call of add_feat
        all_pairs = []
        logging.info("Beginning feature evaluations on transformed features")
        for s in sorteds[0:K]:
            # check if num2num, cat2cat, or cat2num
            pair = interaction_pairs[s]
            el1 = pair[0]
            el2 = pair[1]
            print(f"el1: {el1}")
            print(f"el1: {el2}")
            print(f"size: vartype_list {len(vartype_list)}")
            if vartype_list[el1]=="num" and vartype_list[el2]=="num":
                combo_type = "numnum"
            elif vartype_list[el1]=="num" and (vartype_list[el2]=="cat" or vartype_list[el2]=="ord"):
                combo_type = "numcat"
            elif (vartype_list[el1]=="cat" or vartype_list[el1]=="ord") and vartype_list[el2]=="num":
                combo_type = "catnum"
            else:
                combo_type = "catcat"
            all_pairs.append((el1,el2,combo_type))

        X=(X_train.copy(),X_test.copy())
        future_feats = [add_features(combo_type, op, X, el1, el2) for el1,el2,combo_type in all_pairs for op in ops(combo_type)]
        num_new=0
        for f in future_feats:
            list_X_train.append(f[0])
            list_X_test.append(f[1])
            list_trans_strings.append(f[2])
            list_feat_pairs.append(f[3])
            list_types.append(f[4])
            num_new+=1
        future_scores = [eval_features.remote(clf,list_X_train[i].copy(), y_train.copy(), scoring, list_trans_strings[i], model, vartype_list, extra_type=list_types[i]) for i in range(0,num_new)]
        future_scores_out=ray.get(future_scores)
        for f in future_scores_out:
            list_scores.append(f[0])
            list_feats.append(f[1])
        #find the best addition
        if len(list_scores)>0:
            top_valscore = np.argmax(np.array(list_scores))
            logging.info(f"Feature added: {list_trans_strings[top_valscore]}")
        
        type_added=None
        if len(list_scores)>0:
            loc = list_trans_strings.index(list_feats[top_valscore])
            X_train = list_X_train[loc].copy()
            X_test = list_X_test[loc].copy()
            if not list_trans_strings[loc]=="NONE":
                vartype_list.append(list_types[loc])
                is_feat_added=True
            else:
                is_feat_added=False
            explored_pairs.append(list_feat_pairs[loc])
            popspot = interaction_pairs.index(list_feat_pairs[loc])
            operation_list.append(list_trans_strings[loc])
            logging.info("Feature pair removed from exploration: " + str(interaction_pairs.pop(popspot)))
            interaction_information.pop(popspot)
            type_added=list_types[loc]

        #now find best unary transform on recent feature
        if not model == "RF" and not model == "RFR" and type_added=="num":
            X_train, X_test = unary(clf, X_train, y_train, X_test, scoring, vartype_list, seed=seed2*100 + it+1)

        # handle any infs
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        if model == "LR" or model == "lasso":
            # one hot encode
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
        
        clf.fit(X_train_temp,y_train)

        feat_length.append(X_train.shape[1])

        cv = val_score(clf,scoring,X_train_temp,y_train,vartype_list,seed=seed2*100 + it+1)
        logging.info("val score of new feats: " + str(cv))
        #see if added features are better than last iter
        cvs.append(cv)
    
        
        if len(stored_cv)<n_lags:
            stored_cv.append(cv)
        else:
            stored_cv.pop(0)
            stored_cv.append(cv)
            # avg_inc = np.mean(np.array(stored_cv[1:n_lags])-np.array(stored_cv[0:n_lags-1]))
            avg_inc = np.mean(np.array(stored_cv[n_lags//2:]))-np.mean(np.array(stored_cv[0:n_lags//2]))
            logging.info("Average increase in score: " + str(avg_inc))
            if avg_inc<=eps:
                break


    bigend1 = time.time()

    logging.info("Time spent: " + str(bigend1 - bigstart1))

    feats_added = X_train.shape[1] - original_feature_size
    # remove final feature added that cause val score to potentially drop
    return X_train[:,:-1], y_train, X_test[:,:-1], vartype_list[:-1], cvs, operation_list

