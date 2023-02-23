import pandas as pandas_package
from sklearn.linear_model import LinearRegression
# pickle package
import pickle as python_pickle_module

# Step-1: Prepare data
# a. Read data
df_object = pandas_package.read_csv("../../datasets/export_model/homeprices.csv")
# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()
x_axis_data = df_object[["area"]]
y_axis_data = df_object["price"]
# c. Prepare data used to predict
df_predict_prices = pandas_package.DataFrame([
     {"area": "4400"},
     {"area": "3400"}
])

# Step-2: Create linear regression object
linear_regression_model_object = LinearRegression()

# Step-3: Train the model with existing data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
linear_regression_model_object.fit(x_axis_data, y_axis_data)

# Get slope and intercept
slope, intercept = linear_regression_model_object.coef_, linear_regression_model_object.intercept_

# Step-4: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using single value
predict_4400 = linear_regression_model_object.predict([[4400]])
predict_3400 = linear_regression_model_object.predict([[3400]])

# b. Predict range of values
predict_prices = linear_regression_model_object.predict(df_predict_prices)

# Step-5: Verify the accuracy of the model using score() method
# model_accuracy = linear_regression_model_object.score(df_predict_2017_2023, predict_2017_to_2023)
# print(model_accuracy)

# Step-6: Save trained model
# As pickle file using python pickle module.
with open("save_model_as_pickle", "wb") as file:    # wb stands for write binary data
    python_pickle_module.dump(linear_regression_model_object, file)     # dump() used to store the model in file.
