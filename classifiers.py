import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, \
    confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import log


def print_results(results, classifier):
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        log.log('{} (+/-{}) for {}'.format(round(mean, 4), round(std * 2, 4), params))
    log.log('BEST PARAMS for {}: {}\n'.format(classifier, results.best_params_))


def evaluate_model(name, model, features, labels):
    start = time.time()
    prediction = model.predict(features)
    end = time.time()

    recall = round(recall_score(labels, prediction, pos_label='M'), 4)
    accuracy = round(accuracy_score(labels, prediction), 4)
    precision = round(precision_score(labels, prediction, pos_label='M'), 4)
    f1 = round(f1_score(labels, prediction, pos_label='M'), 4)
    auc = round(roc_auc_score(labels, model.predict_proba(features)[:, 1]), 4)

    cm = confusion_matrix(labels, prediction)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    fall_out = FP[0] / (FP[0]+FN[0])

    log.log('\n{} --  AUC: {} / Recall: {} / Precision: {} / F1-Score: {} / Accuracy: {} / Fall-out: {}'.format(
        name, auc, recall, precision, f1, accuracy, fall_out))
    log.log("{} & {} & {} & {} & {} & {}".format(auc, recall, precision, f1, accuracy, fall_out))


def random_forest(X_train, y_train):
    log.log("TRAINING: Random Forest")

    start_time = time.time()
    rf_classifier = RandomForestClassifier()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
    }

    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print_results(grid_search, "Random Forrest")
    joblib.dump(grid_search.best_estimator_, 'Models/RF_model.pkl')
    log.log("Train time for Random Forrest: " + str((time.time() - start_time) / 60) + " min")

    return grid_search.best_estimator_


def logistic_regression(X_train, y_train):
    log.log("TRAINING: Logistic Regression")
    start_time = time.time()
    lr_classifier = LogisticRegression()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength (smaller values mean stronger regularization)
        'penalty': ['l1', 'l2'],  # Regularization type (L1 or L2)
        'solver': ['liblinear', 'saga'],  # Solver for optimization
    }

    grid_search = GridSearchCV(lr_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print_results(grid_search, "Logistic Regression")
    joblib.dump(grid_search.best_estimator_, 'Models/LR_model.pkl')
    log.log("Train time for Logistic Regression: " + str((time.time() - start_time) / 60) + " min")

    return grid_search.best_estimator_


def decision_tree(X_train, y_train):
    log.log("TRAINING: Decision Tree")

    start_time = time.time()
    dt_classifier = DecisionTreeClassifier()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],  # Split criterion
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    }

    grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print_results(grid_search, "Decision Tree")
    joblib.dump(grid_search.best_estimator_, 'Models/DT_model.pkl')
    log.log("Train time for Decision Tree: " + str((time.time() - start_time) / 60) + " min")

    return grid_search.best_estimator_


def naive_bayes(X_train, y_train):
    # ## Gaussian Naive Bayes
    log.log("TRAINING: Naive Bayes")

    start_time = time.time()
    nb_classifier = GaussianNB()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],  # Smoothing parameter (alpha)
    }

    grid_search = GridSearchCV(nb_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print_results(grid_search, "Naive Bayes")
    joblib.dump(grid_search.best_estimator_, 'Models/NB_model.pkl')
    log.log("Train time for Naive Bayes: " + str((time.time() - start_time) / 60) + " min")

    return grid_search.best_estimator_


def support_vector_machine(X_train, y_train):
    log.log("TRAINING: Support Vector Machine")
    start_time = time.time()
    svm_classifier = SVC()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['sigmoid', 'rbf'],  # Kernel type
        'probability': [True],
        'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient (only for 'rbf' and 'poly')
    }

    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print_results(grid_search, "Support Vector Machine")
    joblib.dump(grid_search.best_estimator_, 'Models/SVM_model.pkl')
    log.log("Train time for Support Vector Machine: " + str((time.time() - start_time) / 60) + " min")

    return grid_search.best_estimator_
