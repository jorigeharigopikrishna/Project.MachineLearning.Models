import pandas as pandas_package
import matplotlib.pyplot as plot_package
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Using built-in hand_written digits dataset
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import seaborn as seaborn_package

# Step-1: Prepare data
# a. Read data from built-in dataset of digits
df_object = load_digits()   # use of handwritten digits from sklearn datasets
print(dir(df_object))   # ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']

# b. Check for any NaN values in dataframe
#any_nan_columns = df_object.isna().sum()

# c. Prepare x and y axis data from dataframe
x_axis_data = df_object["data"]
y_axis_data = df_object["target"]

# d. Prepare data used to predict

# e: Split the available data into train and test data using train_test_split method
# 20% of available dataset to be used for test data
# 80% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.2)

# Step-2: Plot datapoints using scatter plot.
# Make sure that datapoints follow S-shaped curve to use logistic regression model

# To visualize the digit, Plot images of digits
plot_package.gray()     # for background gray
# matshow() means matrix show. Used to display matrices of any order.
# Since each image in digits dataset is 8*8 matrix size, using matshow() to draw image
plot_package.matshow(df_object.images[4])   # Handwritten image of 4
plot_package.show()

# Step-3: Create logistic regression object
logistic_regression_model_object = LogisticRegression()

# Step-4: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
logistic_regression_model_object.fit(X_train, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = logistic_regression_model_object.predict(X_test)

# b. Predict using single value
predict_6 = logistic_regression_model_object.predict([x_axis_data[67]])     # 67 represents 6 digit

# c. Predict range of values
predict_65_to_74_digits = logistic_regression_model_object.predict(x_axis_data[65:75])     # 67 represents 6 digit

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = logistic_regression_model_object.score(X_test, y_test)

# Step-7: Get a confusion matrix to study the behavior of ML model
confusion_matrix = confusion_matrix(y_test, predict_test_data)
# Use seaborn.heatmap() plot to visualize the confusion_matrix in a better way.
seaborn_package.heatmap(confusion_matrix, annot=True)   # Draw a heatmap with confusion matrix.
plot_package.xlabel("Predicted by model")
plot_package.ylabel("Actual value")
plot_package.title("Confusion Matrix")
plot_package.show()

# Step-8: Look at the probability of the predictions using predict_proba() method

# Step-9: Perform other steps like exporting to new csv file, save as pickle.
