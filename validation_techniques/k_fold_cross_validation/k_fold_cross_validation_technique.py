import pandas as pandas_package
import numpy as numpy_package
from sklearn.datasets import load_digits
# Using various models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# Using KFold APIs
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Understand the K-Fold technique using simple dataset.
numbers_dataset = numpy_package.arange(1, 21)   # 20 numbers
k_fold_numbers_object = KFold(n_splits=5)   # Create 5 folds where each fold has 4 elements
dataset_folds = k_fold_numbers_object.split(numbers_dataset)
for fold_train_dataset, fold_test_dataset in dataset_folds:     # Iterate through splits
    print(fold_train_dataset)   # Train folds at each iteration
    print(fold_test_dataset)    # Test fold at each iteration


# Create some common definitions
def get_model_score(model_object, x_train, y_train, x_test, y_test):
    model_object.fit(x_train, y_train)
    model_score = model_object.score(x_test, y_test)
    return model_score

# Step-1: a. Prepare data
digits_dataset = load_digits()

# b. Check for any NaN values in dataframe

# c. prepare x-axis and y-axis data
x_axis_data = digits_dataset.data
y_axis_data = digits_dataset.target

# Step-2: Create model objects to evaluate their performance
lr_model = LogisticRegression()
svm_model = SVC()
rf_model = RandomForestClassifier(n_estimators=40)
gnb_model = GaussianNB()

# Step-3: Store results of model scores for each fold
lr_scores = []
svm_scores = []
rf_scores = []
gnb_scores = []

# Step-4: Using K-Fold API
# Create K-Fold Object
k_fold_object = KFold(n_splits=5)   # 5 folds
k_folds_5 = k_fold_object.split(x_axis_data)    # 5 splits / iterations
for train_dataset_index, test_dataset_index in k_folds_5:   # Iterating through the splits where each iteration will have train dataset index and test dataset index.
    X_train, X_test, y_train, y_test = x_axis_data[train_dataset_index], x_axis_data[test_dataset_index], y_axis_data[train_dataset_index], y_axis_data[test_dataset_index]
    lr_score = get_model_score(lr_model, X_train, y_train, X_test, y_test)
    lr_scores.append(lr_score)
    svm_score = get_model_score(svm_model, X_train, y_train, X_test, y_test)
    svm_scores.append(svm_score)
    rf_score = get_model_score(rf_model, X_train, y_train, X_test, y_test)
    rf_scores.append(rf_score)
    gnb_score = get_model_score(gnb_model, X_train, y_train, X_test, y_test)
    gnb_scores.append(gnb_score)

# Step-5: Get the final score for each model using mean of scores of all folds
lr_final_score = numpy_package.mean(lr_scores)
svm_final_score = numpy_package.mean(svm_scores)
rf_final_score = numpy_package.mean(rf_scores)
gnb_final_score = numpy_package.mean(gnb_scores)

# Step-6: Use of cross_val_score to avoid the steps from Step-3 & Step-4
lr_cross_val_score = cross_val_score(lr_model, x_axis_data, y_axis_data, cv=5)  # Estimator is LogisticRegression()
lr_cross_mean = numpy_package.mean(lr_cross_val_score)
svm_cross_val_score = cross_val_score(svm_model, x_axis_data, y_axis_data, cv=5)    # Estimator is SVC()
svm_cross_mean = numpy_package.mean(svm_cross_val_score)
rf_cross_val_score = cross_val_score(rf_model, x_axis_data, y_axis_data, cv=5)  # Estimator is RandomForestClassifier()
rf_cross_mean = numpy_package.mean(rf_cross_val_score)
gnb_cross_val_score = cross_val_score(gnb_model, x_axis_data, y_axis_data, cv=5)    # Estimator is GuassianNB()
gnb_cross_mean = numpy_package.mean(gnb_cross_val_score)

# using StratifiedKFold API
# Create a StratifiedKFold object
stratified_k_fold_object = StratifiedKFold()
