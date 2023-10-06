import joblib
import pandas as pd
from fast_ml.model_development import train_valid_test_split
import log
import preprocessor


def split_train_test_valid(filename):
    log.log("Reading data from CSV ...")
    df = pd.read_csv(filename, index_col="Unnamed: 0")
    df = preprocessor.data_cleaner(df)

    log.log("SPLITTING DATA")
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target="label", train_size=0.7,
                                                                                valid_size=0.15, test_size=0.15)

    log.log("SAVING DATA")
    joblib.dump(X_train, "DATA/TRAIN/X_train.pkl")
    joblib.dump(y_train, "DATA/TRAIN/y_train.pkl")

    joblib.dump(X_test, "DATA/TEST/X_test.pkl")
    joblib.dump(y_test, "DATA/TEST/y_test.pkl")

    joblib.dump(X_valid, "DATA/VALID/X_valid.pkl")
    joblib.dump(y_valid, "DATA/VALID/y_valid.pkl")
    log.log("----- Done Splitting data -----")

    # return X_train, y_train, X_valid, y_valid, X_test, y_test


def stats(data, labels):
    d = {
        'len': data.shape[0],
        'features': data.shape[1],
        'count': labels.value_counts()
    }
    return d
