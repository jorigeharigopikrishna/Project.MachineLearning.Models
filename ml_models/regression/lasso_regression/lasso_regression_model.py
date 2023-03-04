import pandas as pandas_package
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step-1: Prepare data
# a. Read data
df_object = pandas_package.read_csv("../../../datasets/regression/lasso_regression/housing_prices_melbourne.csv")

# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()
which_columns_have_nan_values = df_object.columns[df_object.isna().any()]

# c. Fill NaN values
# Replace some of columns with 0
columns_replace_nan_with_0 = ['Distance', 'Bedroom2', 'Bathroom', 'Car', 'Propertycount']
df_object[columns_replace_nan_with_0] = df_object[columns_replace_nan_with_0].fillna(0)
# Replace nan values of some of columns with its column mean value
df_object["Landsize"].fillna(df_object["Landsize"].mean(), inplace=True)
df_object["BuildingArea"].fillna(df_object["BuildingArea"].mean(), inplace=True)
# Drop rows of columns where it has atleast 1 NaN value
df_object.dropna(inplace=True)

# d. Convert label text to its equivalent numeric value using get_dummies
# Use of drop_first to drop fist columns in dummy columns to avoid dummy variable trap.
df_object = pandas_package.get_dummies(df_object, drop_first=True)

# e. Prepare x and y axis data
x_axis_data = df_object.drop("Price", axis="columns")
y_axis_data = df_object["Price"]

# f. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.3, random_state=2)

# Step-2: Create linear regression object
linear_regression_model_object = LinearRegression()

# Step-3: Create lasso regression object
lasso_regression_model_object = Lasso(alpha=50, max_iter=100, tol=0.1)

# Step-4: Train the model with existing data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
linear_regression_model_object.fit(X_train, y_train)
lasso_regression_model_object.fit(X_train, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
linear_y_test = linear_regression_model_object.predict(X_test)
lasso_y_test = lasso_regression_model_object.predict(X_test)

# Step-6: Verify the accuracy of the model using score() method
linear_model_accuracy = linear_regression_model_object.score(X_test, y_test)
lasso_model_accuracy = lasso_regression_model_object.score(X_test, y_test)
print(f"Linear model test accuracy: {linear_model_accuracy} and Lasso Model Test accuracy: {lasso_model_accuracy}")
linear_model_train_accuracy = linear_regression_model_object.score(X_train, y_train)
lasso_model_train_accuracy = lasso_regression_model_object.score(X_train, y_train)
print(f"Linear model train accuracy: {linear_model_train_accuracy} and Lasso Model Train accuracy: {lasso_model_train_accuracy}")

# Step-7: Perform other steps like exporting to new csv file.
