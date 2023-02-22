import pandas as pandas_package
import matplotlib.pyplot as plot_package
from sklearn.linear_model import LinearRegression

# Step-1: Prepare data
# a. Read data
df_object = pandas_package.read_csv("../../../datasets/regression/linear_regression/canada_per_capita_income.csv")
x_axis_data = df_object["year"]
y_axis_data = df_object["per capita income (US$)"]
# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()    # sum() function on boolean index array will return columns along with its count of NaN values for each column
# c. Prepare data used to predict
df_predict_2017_2023 = pandas_package.DataFrame([
    {"year": "2017"},
    {"year": "2018"},
    {"year": "2019"},
    {"year": "2020"},
    {"year": "2021"},
    {"year": "2022"},
    {"year": "2023"}
])

# Step-2: Plot datapoints using scatter plot.
# Make sure that datapoints follow linear approach to use linear regression model
plot_package.xlabel("Year")
plot_package.ylabel("Per capita income (US$)")
plot_package.title("Canada year wise per capita income")
plot_package.scatter(x_axis_data, y_axis_data, marker="+", color="red")
plot_package.show()

# Step-3: Create linear regression object
linear_regression_model_object = LinearRegression()

# Step-4: Train the model with existing data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
linear_regression_model_object.fit(df_object[["year"]], df_object["per capita income (US$)"])

# Get the line slope and intercept i.e., y = mx + b
slope, intercept = linear_regression_model_object.coef_, linear_regression_model_object.intercept_

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using single value
predict_2020 = linear_regression_model_object.predict([[2020]])
predict_2023 = linear_regression_model_object.predict([[2023]])

# b. Predict range of values
predict_2017_to_2023 = linear_regression_model_object.predict(df_predict_2017_2023)
df_predict_2017_2023["per capita income (US$)"] = predict_2017_to_2023

# c. Verify the model result using normal maths i.e, y = mx+b
predict_2023_using_maths = slope*2023 + intercept

# Step-6: Verify the accuracy of the model using score() method
#model_accuracy = linear_regression_model_object.score(df_predict_2017_2023, predict_2017_to_2023)
#print(model_accuracy)

# Step-7: Perform other steps like exporting to new csv file.
result_df = pandas_package.concat([df_object, df_predict_2017_2023], ignore_index=True)
result_df.to_csv("../../../datasets/regression/linear_regression/after_predictions_canada_per_capita_income.csv", index=False)

# Visualize the line y=mx+b
plot_package.xlabel("Year")
plot_package.ylabel("Per capita income (US$)")
plot_package.title("Canada year wise per capita income")
plot_package.scatter(df_object.year, df_object["per capita income (US$)"], marker="+", color="red")
plot_package.plot(df_object.year, linear_regression_model_object.predict(df_object[["year"]]), color="green")
plot_package.show()
