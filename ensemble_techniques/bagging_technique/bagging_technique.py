import numpy as numpy_package
import pandas as pandas_package
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# Step-1: Prepare data
df_object = pandas_package.read_csv("../../datasets/bagging/diabetes_details.csv")

# Check for NaN columns
check_nan_columns = df_object.isna().sum()
which_nan_columns = df_object.columns[df_object.isna().any()]

# Original x and y axis data
x_axis_data = df_object.drop("Outcome", axis="columns")
y_axis_data = df_object["Outcome"]

# Split original x and y axis dataset
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.2, random_state=10)

# DecisionTreeClassifier using original x and y axis data
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_model_train_score = tree_model.score(X_train, y_train)
tree_model_test_score = tree_model.score(X_test, y_test)
print(f"Decision Tree Classifier Train score: {tree_model_train_score} and Test score: {tree_model_test_score}")

# RandomForestClassifier using original x and y axis data
forest_model = RandomForestClassifier(n_estimators=10)
forest_model.fit(X_train, y_train)
forest_model_train_score = forest_model.score(X_train, y_train)
forest_model_test_score = forest_model.score(X_test, y_test)
print(f"Random Forest Classifier Train score: {forest_model_train_score} and Test score: {forest_model_test_score}")

# BaggingClassifier using original x and y axis data
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(),
                                  n_estimators=10,
                                  max_samples=0.80,
                                  oob_score=True,
                                  random_state=10)
bagging_model.fit(X_train, y_train)
bagging_model_train_score = bagging_model.score(X_train, y_train)
bagging_model_test_score = bagging_model.score(X_test, y_test)
print(f"Bagging Model Train score: {bagging_model_train_score} and Test score: {bagging_model_test_score}")
