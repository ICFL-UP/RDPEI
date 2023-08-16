import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

data_filename = "16_Ransomware_Detection_Using_PE_Imports.csv"
data = pd.read_csv(data_filename)

# data cleaning
data.drop_duplicates()
data.dropna(how="any", inplace=True)

#  STATS

# stats: number of pe files
print("number of pe files: ", data["SHA256"].nunique())
# stats: number of functions
print("number of functions: ", data["function_name"].nunique())
# stats: number of dll
print("number of dlls: ", data["dll"].nunique())
# stats: number of labels (should be two)
print("number of labels: ", data["label"].nunique())

# preprocessing
print(datetime.now(), ": preprocessing")

# label encoding the functions
label_encoder = preprocessing.LabelEncoder()
data["function_name"] = label_encoder.fit_transform(data["function_name"])
print(data["function_name"].unique())

# label encoding the dll
label_encoder = preprocessing.LabelEncoder()
data["dll"] = label_encoder.fit_transform(data["dll"])
print(data["dll"].unique())

# label encoding the label
print(data["label"].unique())
label_encoder = preprocessing.LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])
print(data["label"].unique())

print(datetime.now(), ": end preprocessing")


# loading the data into variables
array = np.array(data)
y = data["label"]  # extracting the label column to y
x = np.column_stack((data["function_name"], data["dll"]))  # extracting the features

# The scoring metrics dictionary
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1_score': 'f1',
    'area_under_curve': 'roc_auc',
}

# Support Vector Machine
print(datetime.now(), ": TRAINING Support Vector Machine")


# Define the hyperparameter grid for svm
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['sigmoid', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Create GridSearchCV object
svm_grid_search = GridSearchCV(
    SVC(), svm_param_grid, cv=10, scoring=scoring_metrics, refit='accuracy'
)

# Fit the grid search to the points
svm_grid_search.fit(x, y)

# Print the best parameters and the corresponding score
print("Best Parameters:", svm_grid_search.best_params_)
print("Best Scores:")
for metric, score in svm_grid_search.cv_results_['mean_test_recall'].items():
    print(f"{metric}: {score:.4f}")


# Random Forest
print(datetime.now(), ": TRAINING Random Forest")


rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create GridSearchCV object
rf_grid_search = GridSearchCV(
    RandomForestClassifier(), rf_param_grid, cv=10, scoring=scoring_metrics, refit='accuracy'
)

# Fit the grid search to the points
rf_grid_search.fit(x, y)

# Print the best parameters and the corresponding score
print("Best Parameters:", rf_grid_search.best_params_)
print("Best Scores:")
for metric, score in rf_grid_search.cv_results_['mean_test_recall'].items():
    print(f"{metric}: {score:.4f}")

# Logistic Regression
print(datetime.now(), ": TRAINING Logistic Regression")

lr_param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}

# Create GridSearchCV object
lr_grid_search = GridSearchCV(
    LogisticRegression(), lr_param_grid, cv=10, scoring=scoring_metrics, refit='accuracy'
)

# Fit the grid search to the points
lr_grid_search.fit(x, y)

# Print the best parameters and the corresponding score
print("Best Parameters:", lr_grid_search.best_params_)
print("Best Scores:")
for metric, score in lr_grid_search.cv_results_['mean_test_recall'].items():
    print(f"{metric}: {score:.4f}")

# Decision Tree
print(datetime.now(), ": TRAINING Decision Tree")

dt_param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create GridSearchCV object
dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(), dt_param_grid, cv=10, scoring=scoring_metrics, refit='accuracy'
)

# Fit the grid search to the points
dt_grid_search.fit(x, y)

# Print the best parameters and the corresponding score
print("Best Parameters:", dt_grid_search.best_params_)
print("Best Scores:")
for metric, score in dt_grid_search.cv_results_['mean_test_recall'].items():
    print(f"{metric}: {score:.4f}")

# Naive Bayes
print(datetime.now(), ": TRAINING Naive Bayes")

nb_param_grid = {
    # Naive Bayes doesn't have many hyperparameters to tune
}
# Create GridSearchCV object
nb_grid_search = GridSearchCV(
    GaussianNB(), nb_param_grid, cv=10, scoring=scoring_metrics, refit='accuracy'
)

# Fit the grid search to the points
nb_grid_search.fit(x, y)

# Print the best parameters and the corresponding score
print("Best Parameters:", nb_grid_search.best_params_)
print("Best Scores:")
for metric, score in nb_grid_search.cv_results_['mean_test_recall'].items():
    print(f"{metric}: {score:.4f}")



