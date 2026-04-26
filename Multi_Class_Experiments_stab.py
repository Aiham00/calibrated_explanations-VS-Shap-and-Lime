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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from venn_abers import VennAbers
from lime.lime_tabular import LimeTabularExplainer 
from shap import Explainer 
from calibrated_explanations import CalibratedExplainer, __version__

import sys
import os
from os.path import exists, dirname
parent_dir = dirname(os.path.abspath(os.getcwd()))
parent_dir = os.path.join(parent_dir,"VAbme")
sys.path.append(parent_dir)


from data import data_handler
from gru import GRUAttention
sys.setrecursionlimit(3000)
warnings.filterwarnings("ignore")

print("calibrated_explanations V",__version__)
# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)

# ------------------------------------------------------
def test_stability():
    test_size = 20 # number of test samples per dataset
    is_debug = True
    num_rep = 30

    descriptors = ['uncal','va',]#,'va'
    models = ['xGB', 'RF', 'DT', "AAGru"]

    # pylint: disable=line-too-long
    datasets = {
        "CIA": data_handler.load_CIA(),
        "AI4I": data_handler.load_ai4i(),
        "Azure": data_handler.load_azure(True),
        "Pump": data_handler.load_pump()
    }
    tic_all = time.time()

    # -----------------------------------------------------------------------------------------------------
    results = {'num_rep': num_rep, 'test_size': test_size}
    for dataset_name, dataset in datasets.items():

        tic_data = time.time()
        Xn, y, feature_names, label_names = dataset
        X_aagru, _, _, y, _, _, X, _, _, _, _ = data_handler.prepare_data_and_split(Xn, y, test_size=0, cal_size=0)

        no_of_classes = len(np.unique(y))
        no_of_features = Xn.shape[1]
        # Define Models
        model_dict = {
            'xGB': (xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss'),
                    xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss'),
                    'xGB', X),
            'RF': (RandomForestClassifier(n_estimators=100),
                RandomForestClassifier(n_estimators=100),
                'RF',  X),
            'DT': (DecisionTreeClassifier(),
                DecisionTreeClassifier(),
                    'DT', X),
            'AAGru': (GRUAttention(input_size=X_aagru.shape[2], hidden_size=64, num_classes=no_of_classes),
                    GRUAttention(input_size=X_aagru.shape[2], hidden_size=64, num_classes=no_of_classes),
                    'AAGru',  X)
        }



        model_struct = [model_dict[model] for model in models]
        results[dataset_name] = {}

        for _, c2, alg, X in model_struct:
            tic_algorithm = time.time()
            debug_print(dataset_name+' '+alg)
            results[dataset_name][alg] = {}

            calibrators = {}
            for desc in descriptors:
                calibrators[desc] = {'ce': []}
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42)
            X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train, test_size=0.33,random_state=42)
            if alg == "AAGru":
                X_prop_train = X_prop_train.reshape(X_prop_train.shape[0], 1, X_prop_train.shape[1])
    
            c2.fit(X_prop_train, y_prop_train)

            calibrators['uncal']['model'] = c2
            if 'va' in descriptors:
                calibrators['va']['model'] = VennAbers()
                calibrators['va']['model'].fit(c2.predict_proba(X_cal), y_cal)
            calibrators['data'] = {'X_prop_train':X_prop_train,'y_prop_train':y_prop_train,'X_cal':X_cal,'y_cal':y_cal,'X_test':X_test,'y_test':y_test,}

            np.random.seed(1337)
            categorical_features = [i for i in range(no_of_features) if len(np.unique(Xn.iloc[:,i])) < 10]

            ce = CalibratedExplainer(c2, X_cal, y_cal, \
                    feature_names=feature_names,class_labels= label_names)

            lime = LimeTabularExplainer(X_cal, feature_names=feature_names,mode='classification', random_state=42)
            shap = Explainer(lambda x: calibrators['uncal']['model'].predict_proba(x)[:,1], X_cal, \
                feature_names=feature_names)
            shap.random_state = 42
            shap_va = Explainer(lambda x: calibrators['va']['model'].predict_proba(x)[0][:,1], X_cal, \
                        feature_names=feature_names)
            shap_va.random_state = 42

            stability =  {'ce':[], 'cce':[], 'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}
            stab_timer = {'ce':[], 'cce':[], 'lime':[], 'lime_va':[], 'shap':[], 'shap_va':[]}

            i = 0
            while i < num_rep:
                try:
                    ce.seed = i
                    tic = time.time()
                        
                    factual_explanations = ce.explain_factual(X_test, multi_lables_explanation = True)

                    ct = time.time()-tic
                    stab_timer['ce'].append(ct)

                    stability['ce'].append([[f.feature_weights for f in e.values()] for e in factual_explanations.explanations])

                    ce.seed = i
                    tic = time.time()
                    con_factual_explanation = ce.explore_alternatives(X_test, multi_lables_explanation = True)
                    ct = time.time()-tic
                    stab_timer['cce'].append(ct)
                    stability['cce'].append([[f.feature_weights for f in e.values()] for e in con_factual_explanation.explanations])

                    model = calibrators['uncal']['model']
                    tic = time.time()
                    lime_exps_cl = []
                    for instance in X_test:
                        exp = lime.explain_instance(instance, model.predict_proba,  top_labels=no_of_classes)
                        lime_exps_cl.append(exp)
                    ct = time.time()-tic
                    stab_timer['lime'].append(ct)

                    tmps = []
                    for j in range(len(lime_exps_cl)):
                        tmp = np.zeros(no_of_features)
                        for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                            tmp[f[0]] = f[1]
                        tmps.append(tmp)
                    stability['lime'].append(tmps)

                    model = calibrators['va']['model']
                    tic = time.time()
                    lime_exps_cl = []
                    for instance in X_test:
                        exp = lime.explain_instance(instance, lambda x: model.predict_proba(x)[0],  top_labels=no_of_classes)
                        lime_exps_cl.append(exp)
                    ct = time.time()-tic
                    stab_timer['lime_va'].append(ct)

                    tmps = []
                    for j in range(len(lime_exps_cl)):
                        tmp = np.zeros(no_of_features)
                        for _,f in enumerate(lime_exps_cl[j].local_exp[1]):
                            tmp[f[0]] = f[1]
                        tmps.append(tmp)
                    stability['lime_va'].append(tmps)
                    tic = time.time()
                    explanations = shap(X_test)
                    ct = time.time()-tic
                    stab_timer['shap'].append(ct)

                    stability['shap'].append(explanations.values) # pylint: disable=no-member

                    tic = time.time()
                    explanations = shap_va(X_test)
                    ct = time.time()-tic
                    stab_timer['shap_va'].append(ct)
                    # print(f'{ct:.1f}')
                    stability['shap_va'].append(explanations.values) # pylint: disable=no-member
                    i += 1
                except Exception as e: # pylint: disable=broad-exception-caught
                    print(e)
                    warnings.warn(f'Error: {e}')

            results[dataset_name][alg]['stability'] = stability
            results[dataset_name][alg]['stab_timer'] = stab_timer

        toc_data = time.time()
        debug_print(dataset_name + ': ' +str(toc_data-tic_data),is_debug )
        with open('evaluation/results_stab_all_' +dataset_name+'.pkl', 'wb') as f:
            pickle.dump(results, f)

    toc_all = time.time()
    debug_print(str(toc_data-tic_data),is_debug )
test_stability()