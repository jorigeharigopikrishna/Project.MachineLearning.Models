import pandas as pandas_package
from sklearn.linear_model import LinearRegression

# Step-1: Prepare data
df_object = pandas_package.read_csv("../../datasets/validation/car_prices.csv")
# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()
# c. prepare x-axis and y-axis data
x_axis_data = df_object[["Mileage", "Age(yrs)"]]
y_axis_data = df_object["Sell Price($)"]
# d. Prepare data used to predict
df_test_data_predict_future_prices = pandas_package.DataFrame([
      {"Mileage": "50000", "Age(yrs)": "4"},
      {"Mileage": "65000", "Age(yrs)": "6"},
])

# Step-3: Create model object
# Using LinearRegression ML model for practice.
linear_regression_model_object = LinearRegression()

# Step-4: Train the model with existing data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
linear_regression_model_object.fit(x_axis_data, y_axis_data)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using single value
predict_50000_4 = linear_regression_model_object.predict([[50000, 4]])
predict_65000_6 = linear_regression_model_object.predict([[65000, 6]])

# b. Predict range of values
predict_prices = linear_regression_model_object.predict(df_test_data_predict_future_prices)
df_test_data_predict_future_prices["Sell price($)"] = predict_prices

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = linear_regression_model_object.score(df_test_data_predict_future_prices[["Mileage", "Age(yrs)"]], predict_prices)
