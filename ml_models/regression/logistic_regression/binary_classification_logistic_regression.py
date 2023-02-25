import pandas as pandas_package
import matplotlib.pyplot as plot_package
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step-1: Prepare data
# a. Read data
df_object = pandas_package.read_csv("../../../datasets/regression/logistic_regression/insurance_data.csv")
# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()
# c. Prepare x and y axis data from dataframe
x_axis_data = df_object[["age"]]
y_axis_data = df_object["bought_insurance"]
# d. Prepare data used to predict
df_predict_insurance_bought_or_not = pandas_package.DataFrame([
     {"age": "20"},
     {"age": "50"},
     {"age": "35"},
     {"age": "65"}
])
# e: Split the available data into train and test data using train_test_split method
# 10% of available dataset to be used for test data
# 90% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.1)

# Step-2: Plot datapoints using scatter plot.
# Make sure that datapoints follow S-shaped curve to use logistic regression model
plot_package.xlabel("Age")
plot_package.ylabel("Insurance bought or not")
plot_package.title("Insurance vs Age")
plot_package.scatter(x_axis_data, y_axis_data, marker="+", color="red")
plot_package.show()

# Step-3: Create logistic regression object
# Categorical value of 0 indicates No
# Categorical value of 1 indicates Yes
logistic_regression_model_object = LogisticRegression()

# Step-4: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
logistic_regression_model_object.fit(X_train, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = logistic_regression_model_object.predict(X_test)

# b. Predict using single value
predict_45 = logistic_regression_model_object.predict([[45]])   # Should return 1
predict_15 = logistic_regression_model_object.predict([[15]])   # Should return 0

# c. Predict range of values
predict_insurance_bought = logistic_regression_model_object.predict(df_predict_insurance_bought_or_not)

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = logistic_regression_model_object.score(X_test, y_test)
model_accuracy_new = logistic_regression_model_object.score(df_predict_insurance_bought_or_not, [0, 1, 0, 1])

# Step-7: Look at the probability of the predictions using predict_proba() method
# It will return two values for each predicted value, left value indicates probability for 0 and right value indicates probability for 1
predict_probability = logistic_regression_model_object.predict_proba(df_predict_insurance_bought_or_not)

# Step-7: Perform other steps like exporting to new csv file, save as pickle.
# result_df = pandas_package.concat([df_object, df_predict_2017_2023], ignore_index=True)
# result_df.to_csv("../../../datasets/regression/linear_regression/after_predictions_canada_per_capita_income.csv", index=False)
