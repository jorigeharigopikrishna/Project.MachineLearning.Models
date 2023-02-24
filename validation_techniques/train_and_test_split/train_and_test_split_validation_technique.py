import pandas as pandas_package
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step-1: Prepare data
df_object = pandas_package.read_csv("../../datasets/validation/car_prices.csv")
# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()
# c. prepare x-axis and y-axis data
x_axis_data = df_object[["Mileage", "Age(yrs)"]]
y_axis_data = df_object["Sell Price($)"]

# Step-2: Prepare train data and test data using train_test_split
# a. Default test_size is 0.25 and random pick of train data and test data from dataset
# X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data)

# b. test_size is 0.20 and random pick of train data and test data from dataset
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.20)

# c. test_size is 0.10 and constant pick of train data and test data from dataset
# X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.10, random_state=2)

# Step-3: Create model object
# Using LinearRegression ML model for practice.
linear_regression_model_object = LinearRegression()

# Step-4: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
linear_regression_model_object.fit(X_train, y_train)

# Step-5: Start predictions and test the model with test data using the predict() method
# predict(x_2D_array)
predict_prices = linear_regression_model_object.predict(X_test)

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = linear_regression_model_object.score(X_test, y_test)
