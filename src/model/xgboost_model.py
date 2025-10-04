
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import datetime
# import packages for hyperparameters tuning
from sklearn.metrics import accuracy_score

def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)

def invokeXGBoost(train_set_x, train_set_y, test_set_x, test_set_y, test_set_id):
    train_set_x = np.asarray(train_set_x)
    test_set_x = np.asarray(test_set_x)
    #class_weight = np.count_nonzero(train_set_y)/(len(train_set_y)-np.count_nonzero(train_set_y))
    class_weight = 0.9
    model = XGBClassifier(eta=0.1, n_estimators=4000, max_depth=7, scale_pos_weight=class_weight, objective='binary:logistic')
    #model = XGBClassifier(colsample_bytree=0.5626804365643123, gamma=8.851525057093188, max_depth=4, min_child_weight=4, reg_alpha=87, reg_lambda=0.9289003759734173)
    model.fit(train_set_x, train_set_y)
    print("\r\n")

    # evaluate the model on the test set
    print("predicting on the test set")

    y_predict = model.predict(test_set_x)
    y_predict_proba = model.predict_proba(test_set_x)

    accuracy = np.mean(test_set_y == y_predict) * 100
    print("accuracy = " + str(accuracy))

    target_names = ["Non-vulnerable", "Vulnerable"]  # non-buggy->0, buggy->1
    print(confusion_matrix(test_set_y, y_predict, labels=[0, 1]))
    print("\r\n")
    print("\r\n")
    print(classification_report(test_set_y, y_predict, target_names=target_names))

    #import joblib
    #joblib.dump(model, 'xgboost_model.pkl')

    if not isinstance(y_predict_proba, list): probs = y_predict_proba.tolist()
    if not isinstance(test_set_id, list): test_id = np.asarray(test_set_id).tolist()
    zippedlist = list(zip(test_set_id, y_predict_proba, test_set_y))
    result_set = pd.DataFrame(zippedlist, columns=['Function_ID', 'Probs. of being vulnerable', 'Label'])
    ListToCSV(result_set, 'GNN_network_XGBoost' + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + '_result.csv')

