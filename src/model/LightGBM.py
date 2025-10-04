
import numpy as np
import lightgbm as lgb
import pandas as pd
import datetime
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)

#def invokeLightGBM(train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y, test_set_id):
def invokeLightGBM(train_set_x, train_set_y, test_set_x, test_set_y, test_set_id):
    """
    lgb_clf = lgb.LGBMClassifier(boosting_type='dart', num_leaves=1200, max_depth=-1, learning_rate=0.05, n_estimators=4000,
                        subsample_for_bin=200000, objective='binary', class_weight='balanced', min_split_gain=0.0,
                       min_child_weight=0.001, min_child_samples=100, subsample=1, subsample_freq=0,
                        colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=39, n_jobs=-1,
                        importance_type='split')
    Best configs:
    lgb_clf = lgb.LGBMClassifier(
        boosting_type='dart',
        objective='binary',
        scale_pos_weight=32,  # 关键调整点
        class_weight={0: 1.0, 1: 5.0},
        # 树结构控制
        num_leaves=31,
        max_depth=5,
        min_child_samples=50,
        min_split_gain=0.1,
        # 正则化
        reg_alpha=0.5,
        reg_lambda=1.0,
        # 随机性
        subsample=0.7,
        colsample_bytree=0.5,
        # 学习策略
        learning_rate=0.04,
        n_estimators=8000,
    """
    lgb_clf = lgb.LGBMClassifier(
        boosting_type= 'gbdt',  # GBDT often works better than DART for imbalanced data
        objective='binary',
        metric= 'None',  # We'll use custom eval metric
        scale_pos_weight=67,  # Use actual class ratio instead of 36
        class_weight=None,  # Don't use both scale_pos_weight and class_weight
        #is_unbalance=True,  # LightGBM's built-in handling

        # Tree structure - more conservative to prevent overfitting
        num_leaves= 31,
        max_depth=6,
        min_child_samples=100,  # Increased to prevent overfitting
        min_split_gain=0.2,
        min_child_weight= 0.001,

        # Regularization
        reg_alpha=1.0,
        reg_lambda=2.0,
        bagging_fraction=0.8,
        feature_fraction=0.8,
        bagging_freq=5,

        # Learning strategy
        learning_rate=0.03,  # Lower learning rate
        n_estimators= 10000,
        #early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        verbose= -1
    )
    train_set_x = np.asarray(train_set_x)
    train_set_y = np.ndarray.flatten(np.asarray(train_set_y))
    #validation_set_x = np.asarray(validation_set_x)
    #validation_set_y = np.ndarray.flatten(np.asarray(validation_set_y))
    test_set_x = np.asarray(test_set_x)
    test_set_y = np.ndarray.flatten(np.asarray(test_set_y))

    train_x_shuffled, train_y_shuffled = shuffle(train_set_x, train_set_y, random_state=42)

    clf = lgb_clf.fit(train_x_shuffled, train_y_shuffled)

    # evaluate the model on the test set
    print("predicting on the test set")
    # t0 = time()
    y_predict = clf.predict(test_set_x)

    y_predict_proba = clf.predict_proba(test_set_x)

    # Accuracy
    accuracy = np.mean(test_set_y == y_predict) * 100
    print("accuracy = " + str(accuracy))

    target_names = ["Non-vulnerable", "Vulnerable"]  # non-buggy->0, buggy->1
    print(confusion_matrix(test_set_y, y_predict, labels=[0, 1]))
    print("\r\n")
    print("\r\n")
    print(classification_report(test_set_y, y_predict, target_names=target_names))

    #import joblib
    #joblib.dump(clf, 'lightgbm_model.pkl')

    if not isinstance(y_predict_proba, list): probs = y_predict_proba.tolist()
    if not isinstance(test_set_id, list): test_id = np.asarray(test_set_id).tolist()
    zippedlist = list(zip(test_set_id, y_predict_proba, test_set_y))
    result_set = pd.DataFrame(zippedlist, columns=['Function_ID', 'Probs. of being vulnerable', 'Label'])
    ListToCSV(result_set, 'WAGGNN_LightGBM_rw' + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + '_result_best.csv')
