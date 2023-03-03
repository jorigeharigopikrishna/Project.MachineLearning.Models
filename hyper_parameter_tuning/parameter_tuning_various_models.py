import pandas as pandas_package
# Digits dataset
from sklearn.datasets import load_digits
# Various models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Grid Search CV
from sklearn.model_selection import GridSearchCV
# Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV

# Step-1: Prepare data
digits_dataset = load_digits()

# Step-2: Prepare x and y axis data
x_axis_data = digits_dataset.data
y_axis_data = digits_dataset.target

# Tune parameters manually by invoking the model multiple times
# Using this manually technique to perform a lot of combinations is not a good appraoch
# Logistic Regression model
lr_c_5 = LogisticRegression(C=5)
lr_c_10 = LogisticRegression(C=10)
lr_c_15 = LogisticRegression(C=15)
# Random Forest Classifier model
random_forest_10_trees = RandomForestClassifier(n_estimators=10)
random_forest_25_trees = RandomForestClassifier(n_estimators=25)
random_forest_50_trees = RandomForestClassifier(n_estimators=50)
# SVC model
svc_c_10_kernel_linear = SVC(C=10, kernel="linear", gamma="auto")
svc_c_10_kernel_rbf = SVC(C=10, kernel="rbf", gamma="auto")
svc_c_20_kernel_linear = SVC(C=20, kernel="linear", gamma="auto")
svc_c_20_kernel_rbf = SVC(C=20, kernel="rbf", gamma="auto")

# Using GridSearchCV to perform various combinations
# Logistic Regression with C values of 5-15
grid_search_lr = GridSearchCV(LogisticRegression(), {
    "C": [5, 10, 15]
}, cv=5, return_train_score=False)
grid_search_lr.fit(x_axis_data, y_axis_data)
lr_df = pandas_package.DataFrame(grid_search_lr.cv_results_)
lr_best_score = grid_search_lr.best_score_
lr_best_params = grid_search_lr.best_params_
# Random Forest Classifier with 10-50 trees
grid_search_rf = GridSearchCV(RandomForestClassifier(), {
    "n_estimators": [10, 20, 30, 40, 50]
}, cv=5, return_train_score=False)
grid_search_rf.fit(x_axis_data, y_axis_data)
rf_df = pandas_package.DataFrame(grid_search_rf.cv_results_)
rf_best_score = grid_search_rf.best_score_
rf_best_params = grid_search_rf.best_params_
# SVC with C and kernel values
grid_search_svc = GridSearchCV(SVC(), {
    "C": [10, 20, 30],
    "kernel": ["linear", "rbf"]
}, cv=5, return_train_score=False)
grid_search_svc.fit(x_axis_data, y_axis_data)
svc_df = pandas_package.DataFrame(grid_search_svc.cv_results_)
svc_best_score = grid_search_svc.best_score_
svc_best_params = grid_search_svc.best_params_

# Using RandomizedSearchCV to perform various combinations but not all possible combinations
# Logistic Regression with C values of 5-15
randomized_search_lr = RandomizedSearchCV(LogisticRegression(), {
    "C": [5, 10, 15]
}, cv=5, return_train_score=False, n_iter=3)
randomized_search_lr.fit(x_axis_data, y_axis_data)
lr_rs_df = pandas_package.DataFrame(randomized_search_lr.cv_results_)
lr_rs_best_score = randomized_search_lr.best_score_
lr_rs_best_params = randomized_search_lr.best_params_
# Random Forest Classifier with 10-50 trees - pick only 3 combinations
randomized_search_rf = RandomizedSearchCV(RandomForestClassifier(), {
    "n_estimators": [10, 20, 30, 40, 50]
}, cv=5, return_train_score=False, n_iter=3)
randomized_search_rf.fit(x_axis_data, y_axis_data)
rf_rs_df = pandas_package.DataFrame(randomized_search_rf.cv_results_)
rf_rs_best_score = randomized_search_rf.best_score_
rf_rs_best_params = randomized_search_rf.best_params_
# SVC with C and kernel values - pick only 3 combinations
randomized_search_svc = RandomizedSearchCV(SVC(), {
    "C": [10, 20, 30],
    "kernel": ["linear", "rbf"]
}, cv=5, return_train_score=False, n_iter=3)
randomized_search_svc.fit(x_axis_data, y_axis_data)
svc_rs_df = pandas_package.DataFrame(randomized_search_svc.cv_results_)
svc_rs_best_score = randomized_search_svc.best_score_
svc_rs_best_params = randomized_search_svc.best_params_
