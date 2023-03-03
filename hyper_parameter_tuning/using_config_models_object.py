import pandas as pandas_package
# Digits dataset
from sklearn.datasets import load_digits
# Various models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Train Test Split
from sklearn.model_selection import train_test_split
# Grid Search CV
from sklearn.model_selection import GridSearchCV
# Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV

# Step-1: Prepare data
digits_dataset = load_digits()

# Step-2: Prepare x and y axis data
x_axis_data = digits_dataset.data
y_axis_data = digits_dataset.target

models_config_object = {
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
           "n_estimators": [10, 20, 30, 40, 50]
        }
    },
    "svc": {
        "model": SVC(gamma="auto"),
        "params": {
            "C": [10, 20, 30],
            "kernel": ["linear", "rbf"]
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(),
        "params": {
           "C": [5, 10, 15]
        }
    },
}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.3, random_state=2)

# Using GridSearchCV to perform various combinations
for config_name, config_values in models_config_object.items():
    grid_search_cv_object = GridSearchCV(config_values["model"], config_values["params"], cv=5, return_train_score=False)
    grid_search_cv_object.fit(X_train, y_train)
    print("Using Grid Search CV")
    print(f"Model: {config_values['model']} - Best Score: {grid_search_cv_object.best_score_} - Best Params: {grid_search_cv_object.best_params_}")

# Using RandomizedSearchCV to perform only 3 random combinations
for config_name, config_values in models_config_object.items():
    grid_search_cv_object = RandomizedSearchCV(config_values["model"], config_values["params"], cv=5, return_train_score=False, n_iter=3)
    grid_search_cv_object.fit(X_train, y_train)
    print("Using Randomized Search CV")
    print(f"Model: {config_values['model']} - Best Score: {grid_search_cv_object.best_score_} - Best Params: {grid_search_cv_object.best_params_}")
