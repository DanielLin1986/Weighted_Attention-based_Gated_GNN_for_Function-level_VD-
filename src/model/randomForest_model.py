
import datetime
import pickle
import numpy as np
from numpy import genfromtxt
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)

def invokeRandomForest(train_set_x, train_set_y, test_set_x, test_set_y, test_set_id):
    #    param_grid = {'max_depth': [15,20,25,30],
    #                  'min_samples_split': [4,5,6],
    #                  'min_samples_leaf': [2,3,4,5],
    #                  'bootstrap': [True,False],
    #                  'criterion': ['gini','entropy'],
    #                  'n_estimators': [40,50,55,60,65]}
    #
    train_set_y = np.ndarray.flatten(np.asarray(train_set_y))
    test_set_y = np.ndarray.flatten(np.asarray(test_set_y))

    # clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)

    clf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                                 criterion='entropy', max_depth=40,
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=3,
                                 min_samples_split=4, min_weight_fraction_leaf=0.0,
                                 n_estimators=8000, oob_score=False, random_state=39,
                                 verbose=1, warm_start=False, n_jobs=-1)
    """
    clf = RandomForestClassifier(verbose=1, n_jobs=-1, n_estimators=8000, random_state=39)
    """
    clf = clf.fit(train_set_x, train_set_y)

    print("feature importance:")
    print(clf.feature_importances_)
    print ("\n")

    # print("best estimator found by grid search:")
    # print(clf.best_estimator_)

    print ("\r\n")

    # evaluate the model on the test set
    print("predicting on the test set")
    # t0 = time()
    y_predict = clf.predict(test_set_x)

    y_predict_proba = clf.predict_proba(test_set_x)

    # Accuracy
    accuracy = np.mean(test_set_y==y_predict) *100
    print ("accuracy = " + str(accuracy))

    target_names = ["Non-vulnerable","Vulnerable"]  # non-buggy->0, buggy->1
    print (confusion_matrix(test_set_y, y_predict, labels=[0 ,1]))
    print ("\r\n")
    print ("\r\n")
    print (classification_report(test_set_y, y_predict, target_names=target_names))

    if not isinstance(y_predict_proba, list): probs = y_predict_proba.tolist()
    if not isinstance(test_set_id, list): test_id = np.asarray(test_set_id).tolist()
    zippedlist = list(zip(test_set_id, y_predict_proba, test_set_y))
    result_set = pd.DataFrame(zippedlist, columns=['Function_ID', 'Probs. of being vulnerable', 'Label'])
    ListToCSV(result_set, 'GNN_network_RandomForest——new_result' + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + '_result.csv')
