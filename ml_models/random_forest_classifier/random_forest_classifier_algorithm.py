import pandas as pandas_package
import matplotlib.pyplot as plot_package
from sklearn.ensemble import RandomForestClassifier
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

# e. Split the available data into train and test data using train_test_split method
# 20% of available dataset to be used for test data
# 80% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.2)

# Step-2: Create random forest classifier object
# Using default values of n_estimators=10 and criterion=gini
random_forest_classifier_model_object = RandomForestClassifier()

# Use of n_estimators parameter to define the number of trees to be formed.
# random_forest_classifier_model_object = RandomForestClassifier(n_estimators=40)

# Use of criterion parameter which accepts the values like gini, entropy, etc...
# entropy gives better accuracy than gini
# random_forest_classifier_model_object = RandomForestClassifier(criterion="entropy")

# Use of n_estimators and criterion parameters for better accuracy of the model.
# random_forest_classifier_model_object = RandomForestClassifier(n_estimators=40, criterion="entropy")

# Step-3: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
random_forest_classifier_model_object.fit(X_train, y_train)

# Step-4: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = random_forest_classifier_model_object.predict(X_test)

# b. Predict using single value
predict_6 = random_forest_classifier_model_object.predict([x_axis_data[67]])     # 67 represents 6 digit

# c. Predict range of values
predict_65_to_74_digits = random_forest_classifier_model_object.predict(x_axis_data[65:75])     # 67 represents 6 digit

# Step-5: Verify the accuracy of the model using score() method
model_accuracy = random_forest_classifier_model_object.score(X_test, y_test)

# Step-6: Get a confusion matrix to study the behavior of ML model
confusion_matrix = confusion_matrix(y_test, predict_test_data)
# Use seaborn.heatmap() plot to visualize the confusion_matrix in a better way.
seaborn_package.heatmap(confusion_matrix, annot=True)   # Draw a heatmap with confusion matrix.
plot_package.xlabel("Predicted by model")
plot_package.ylabel("Actual value")
plot_package.title("Confusion Matrix")
plot_package.show()

# Step-7: Look at the probability of the predictions using predict_proba() method

# Step-8: Perform other steps like exporting to new csv file, save as pickle.
