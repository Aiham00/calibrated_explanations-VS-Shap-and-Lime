# pylint: disable=invalid-name, line-too-long, duplicate-code
"""
Experiment used in the introductory paper to evaluate the stability and robustness of the explanations
"""
import time
import warnings
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
#
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from venn_abers import VennAbers
from calibrated_explanations import CalibratedExplainer
from lime.lime_tabular import LimeTabularExplainer 
from shap import Explainer 

import sys
import os
from os.path import exists, dirname
parent_dir = dirname(os.path.abspath(os.getcwd()))
parent_dir = os.path.join(parent_dir,"VAbme")
sys.path.append(parent_dir)
from data import data_handler
from gru import GRUAttention
sys.setrecursionlimit(3000)
#warnings.filterwarnings("ignore")

# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# ------------------------------------------------------
def test_robustness():
    test_size = 20 # number of test samples per dataset
    is_debug = True
    num_rep = 30

    descriptors = ['uncal','va',]#,'va'
    models = ['xGB', 'RF', 'DT', "AAGru"]

    # pylint: disable=line-too-long
    datasets = {
        "CIA": data_handler.load_CIA(),
        "AI4I": data_handler.load_ai4i(),
        "Azure": data_handler.load_azure(),
        "Pump": data_handler.load_pump()
    }
    tic_all = time.time()

    # -----------------------------------------------------------------------------------------------------
    results = {'num_rep': num_rep, 'test_size': test_size}
    for dataset_name, dataset in datasets.items():

        tic_data = time.time()
        Xn, y, feature_names, label_names = dataset
        X_aagru, _, _, y, _, _, X, _, _, num_clansses, _ = data_handler.prepare_data_and_split(Xn, y, test_size=0, cal_size=0)

        no_of_classes = len(np.unique(y))
        no_of_features = Xn.shape[1]
        no_of_instances = Xn.shape[0]
        # Define Models

        results[dataset_name] = {}

        for alg in models:
            debug_print(dataset_name)
            results[dataset_name][alg] = {}

            calibrators = {}
            for desc in descriptors:
                calibrators[desc] = {'ce': []}
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42)
            X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=0.33,random_state=42)

            calibrators['data'] = {'X_prop_train':X_prop_train,'y_prop_train':y_prop_train,'X_cal':X_cal,'y_cal':y_cal,'X_test':X_test,'y_test':y_test,}

            robustness = {'ce':[], 'cce':[], 'proba':[],'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[], 'proba_va':[]}
            rob_timer =  {'ce':[], 'cce':[], 'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
            results[dataset_name][alg]['robustness'] = robustness
            results[dataset_name][alg]['rob_timer'] = rob_timer
        i = 0
        while i < num_rep:
            model_dict = {
                'xGB': (xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss',random_state=i),
                        xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss',random_state=i),
                        'xGB', X),
                'RF': (RandomForestClassifier(n_estimators=100,random_state=i),
                    RandomForestClassifier(n_estimators=100,random_state=i),
                    'RF',  X),
                'DT': (DecisionTreeClassifier(random_state=i),
                    DecisionTreeClassifier(random_state=i),
                        'DT', X),
                'AAGru': (GRUAttention(input_size=X_aagru.shape[2], hidden_size=64, num_classes=no_of_classes),
                        GRUAttention(input_size=X_aagru.shape[2], hidden_size=64, num_classes=no_of_classes),
                        'AAGru',  X)
            }

            model_struct = [model_dict[model] for model in models]
            for _, c2, alg, X in model_struct:    
                np.random.seed(i)
                X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=0.33,random_state=i)
                if alg == "AAGru":
                    X_prop_train = X_prop_train.reshape(X_prop_train.shape[0], 1, X_prop_train.shape[1])
                tic_algorithm = time.time()

                c2.fit(X_prop_train,y_prop_train)
                calibrators['uncal']['model'] = c2
                if 'va' in descriptors:
                    calibrators['va']['model'] = VennAbers()
                    calibrators['va']['model'].fit(c2.predict_proba(X_cal), y_cal)
                ce = CalibratedExplainer(c2, X_cal, y_cal, \
                    feature_names=feature_names)
                results[dataset_name][alg]['robustness']['proba'].append(c2.predict_proba(X_test)[:,1])
                results[dataset_name][alg]['robustness']['proba_va'].append(ce.predict_proba(X_test)[:,1])
                try:
                    ce.seed = i
                    tic = time.time()
                    factual_explanations = ce.explain_factual(X_test, multi_lables_explanation = True)

                    ct = time.time()-tic
                    results[dataset_name][alg]['rob_timer']['ce'].append(ct)

                    results[dataset_name][alg]['robustness']['ce'].append([[f.feature_weights for f in e.values()] for e in factual_explanations.explanations])

                    ce.seed = i
                    tic = time.time()
                    con_factual_explanations = ce.explore_alternatives(X_test, multi_lables_explanation = True)
                    ct = time.time()-tic
                    results[dataset_name][alg]['rob_timer']['cce'].append(ct)
                    results[dataset_name][alg]['robustness']['cce'].append([[f.feature_weights for f in e.values()] for e in con_factual_explanations.explanations])

                    lime = LimeTabularExplainer(X_cal, feature_names=feature_names, mode='classification', random_state=i)
                    model = calibrators['uncal']['model']
                    tic = time.time()
                    lime_exps_cl = []
                    for instance in X_test:
                        exp = lime.explain_instance(instance, model.predict_proba, top_labels=no_of_classes)
                        lime_exps_cl.append(exp)
                    ct = time.time()-tic
                    results[dataset_name][alg]['rob_timer']['lime'].append(ct)
                    tmps = []
                    for j in range(len(lime_exps_cl)):
                        tmp = np.zeros(no_of_features)
                        for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                            tmp[f[0]] = f[1]
                        tmps.append(tmp)
                    results[dataset_name][alg]['robustness']['lime'].append(tmps)

                    model = calibrators['va']['model']
                    tic = time.time()
                    lime_exps_cl = []
                    for instance in X_test:
                        exp = lime.explain_instance(instance, lambda x: model.predict_proba(x)[0],top_labels=no_of_classes)
                        lime_exps_cl.append(exp)
                    ct = time.time()-tic
                    results[dataset_name][alg]['rob_timer']['lime_va'].append(ct)
                    tmps = []
                    for j in range(len(lime_exps_cl)):
                        tmp = np.zeros(no_of_features)
                        for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                            tmp[f[0]] = f[1]
                        tmps.append(tmp)
                    results[dataset_name][alg]['robustness']['lime_va'].append(tmps)

                    shap = Explainer(lambda x: calibrators['uncal']['model'].predict_proba(x)[:,1], X_cal, \
                        feature_names=feature_names)
                    # shap.random_state = i
                    tic = time.time()
                    explanations = shap(X_test)
                    ct = time.time()-tic
                    results[dataset_name][alg]['rob_timer']['shap'].append(ct)
                    results[dataset_name][alg]['robustness']['shap'].append(explanations.values) # pylint: disable=no-member

                    shap_va = Explainer(lambda x: calibrators['va']['model'].predict_proba(x)[0][:,1], X_cal, \
                        feature_names=feature_names)
                    # shap.random_state = i
                    tic = time.time()
                    explanations = shap_va(X_test)
                    ct = time.time()-tic
                    results[dataset_name][alg]['rob_timer']['shap_va'].append(ct)
                    results[dataset_name][alg]['robustness']['shap_va'].append(explanations.values) # pylint: disable=no-member
            
                except Exception as e: # pylint: disable=broad-exception-caught
                    warnings.warn(f'Error: {e}')
            i += 1



        toc_data = time.time()
        debug_print(dataset_name + ': ' +str(toc_data-tic_data),is_debug )
        with open('evaluation/results_rob_all_' +dataset_name+'.pkl', 'wb') as f:
            pickle.dump(results, f)

    toc_all = time.time()
    debug_print(str(toc_data-tic_data),is_debug )

#test_robustness()
