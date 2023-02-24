import pandas as pandas_package
import matplotlib.pyplot as plot_package
from sklearn.linear_model import LinearRegression

# Step-1: Prepare data
# a. Read data
df_object = pandas_package.read_csv("../../../datasets/regression/linear_regression/multi_variate_hiring_data.csv")
# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()    # sum() function on boolean index array will return columns along with its count of NaN values for each column
# test_score has NaN values. so replace NaN value with its previous value.
df_object["test_score(out of 10)"].fillna(method="ffill", inplace=True)
# c. prepare x-axis and y-axis data
x_axis_data = df_object[["experience", "test_score(out of 10)", "interview_score(out of 10)"]]
y_axis_data = df_object["salary($)"]
# d. Prepare data used to predict
df_predict_future_salaries = pandas_package.DataFrame([
     {"experience": "2", "test_score(out of 10)": "9", "interview_score(out of 10)": "6"},
     {"experience": "12", "test_score(out of 10)": "10", "interview_score(out of 10)": "10"},
])

# Step-2: Plot datapoints using scatter plot.
# Make sure that datapoints follow linear approach to use linear regression model
plot_package.xlabel("Parameters")
plot_package.ylabel("Salary")
plot_package.title("Salaries of hiring candidates")
plot_package.scatter(df_object["experience"], df_object["salary($)"], marker="+", color="red", label="experience")
plot_package.scatter(df_object["test_score(out of 10)"], df_object["salary($)"], marker="*", color="blue", label="test_score")
plot_package.scatter(df_object["interview_score(out of 10)"], df_object["salary($)"], marker=".", color="green", label="interview_score")
plot_package.legend()
plot_package.show()

# Step-3: Create linear regression object
linear_regression_model_object = LinearRegression()

# Step-4: Train the model with existing data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
linear_regression_model_object.fit(x_axis_data, y_axis_data)

# Get the lines slopes and intercept i.e., y = m1x1 + m2x2 + m3x3 + b
slopes, intercept = linear_regression_model_object.coef_, linear_regression_model_object.intercept_

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using single value
predict_2_9_6 = linear_regression_model_object.predict([[2, 9, 6]])
predict_12_10_10 = linear_regression_model_object.predict([[12, 10, 10]])

# b. Predict range of values
predict_2_9_6_to_12_10_10 = linear_regression_model_object.predict(df_predict_future_salaries)
df_predict_future_salaries["salary($)"] = predict_2_9_6_to_12_10_10  # Create a new column with predicted values.

# c. Verify the model result using normal maths i.e, y = m1x1 + m2x2 + m3x3 + b
predict_2_9_6_using_maths = slopes[0] * 2 + slopes[1] * 9 + slopes[2] * 6 + intercept

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = linear_regression_model_object.score(df_predict_future_salaries[["experience", "test_score(out of 10)", "interview_score(out of 10)"]], predict_2_9_6_to_12_10_10)

# Step-7: Perform other steps like exporting to new csv file.
# result_df = pandas_package.concat([df_object, df_predict_future_salaries], ignore_index=True)
# result_df.to_csv("../../../datasets/regression/linear_regression/after_predictions_hiring_data.csv", index=False)
