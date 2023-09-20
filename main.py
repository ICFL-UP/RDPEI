import traceback

import joblib
import pandas as pd
from datetime import datetime

import classifiers
import log
import preprocessor
import data_reader


def main():
    data_filename = "21_Ransomware_Detection_Using_Features_of_PE_Imports.csv"

    DATA = False
    TRAIN = False
    VALIDATE = False
    TEST = True

    print(datetime.now())

    if DATA:
        # ## CLEANING DATA (done in the data reader when splitting the data)
        # ## DATA SPLIT
        data_reader.split_train_test_valid(data_filename)

    X_train = joblib.load("DATA/TRAIN/X_train.pkl")
    y_train = joblib.load("DATA/TRAIN/y_train.pkl")
    X_valid = joblib.load("DATA/TRAIN/X_valid.pkl")
    y_valid = joblib.load("DATA/TRAIN/y_valid.pkl")
    X_test = joblib.load("DATA/TRAIN/X_test.pkl")
    y_test = joblib.load("DATA/TRAIN/y_test.pkl")

    # ## STATS
    log.log("\nDATASET STATISTICS")
    log.log("Train " + str(data_reader.stats(X_train, y_train)))
    log.log("Validation " + str(data_reader.stats(X_valid, y_valid)))
    log.log("Test " + str(data_reader.stats(X_test, y_test)))

    # ## CLASSIFIER TRAINING
    if TRAIN:
        log.log("\n\nTRAINING CLASSIFIERS")
        try:
            classifiers.random_forest(X_train, y_train)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR TRAINING Random Forest\n")

        try:
            classifiers.decision_tree(X_train, y_train)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR TRAINING Decision Tree\n")

        try:
            classifiers.support_vector_machine(X_train, y_train)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR TRAINING Support Vector Machine\n")

        try:
            classifiers.logistic_regression(X_train, y_train)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR TRAINING Logistic Regression\n")

        try:
            classifiers.naive_bayes(X_train, y_train)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR TRAINING Naive Bayes\n")

        log.log("----- Done training classifiers -----")

    # ## BENCHMARKING THE CLASSIFIERS (With training sets)
    if TRAIN:
        try:
            log.log("\n BENCHMARKING THE CLASSIFIERS")
            log.log("Order of importance: AUC, Recall, Precision, F1 Score, Accuracy, Fall-out")
            models = {}
            for mdl in ['RF', 'LR', 'DT', 'NB', 'SVM']:
                models[mdl] = joblib.load('Models/{}_model.pkl'.format(mdl))
                classifiers.evaluate_model(mdl, models[mdl], X_train, y_train)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR BENCHMARKING THE CLASSIFIERS\n")

    # ## RANKING Random Forest, Decision Tree, Logistic Regression, SVM, NB

    # ## VALIDATION
    if VALIDATE:
        try:
            log.log("\n VALIDATING THE DATASET WITH THE TOP MODELS")
            best_models = {}
            for mdl in ['RF', 'DT', 'LR']:
                best_models[mdl] = joblib.load('Models/{}_model.pkl'.format(mdl))
                classifiers.evaluate_model(mdl, best_models[mdl], X_valid, y_valid)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR IN VALIDATION\n")

    # ## RANK SVM, DT, RF

    # ## TESTING
    if TEST:
        try:
            log.log("\n TESTING WITH THE BEST MODEL... \n")
            best_model = "RF"
            model = joblib.load('Models/{}_model.pkl'.format(best_model))
            classifiers.evaluate_model(best_model, model, X_test, y_test)
        except:
            print(traceback.print_exc())
            log.log("!!! \n\nERROR TESTING\n")


if __name__ == "__main__":
    main()
