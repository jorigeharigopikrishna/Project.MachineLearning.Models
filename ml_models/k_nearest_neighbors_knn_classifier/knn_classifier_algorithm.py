import pandas as pandas_package
import matplotlib.pyplot as plot_package
import seaborn as seaborn_package
# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Using built-in iris flower dataset
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

# Step-1: Prepare data
# a. Read data from built-in dataset of iris flower
iris_dataset = load_iris()   # use of iris flower dataset from sklearn datasets
df_object = pandas_package.DataFrame(iris_dataset["data"], columns=iris_dataset["feature_names"])
df_object["target"] = iris_dataset["target"]

# b. Check for any NaN values in dataframe
# df_object.isna().sum()

# d. Prepare x and y axis data from dataframe
x_axis_data = df_object.drop("target", axis="columns")
y_axis_data = df_object["target"]

# e. Prepare data used to predict

# f. Split the available data into train and test data using train_test_split method
# 20% of available dataset to be used for test data
# 80% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.2)

# Step-3: Create K-Nearest Neighbors (KNN) Classifier object
# Using default values of n_neighbors=5, p=2 and metric="minkowski"
# Use Eucledian Distance (L2) for computation of nearest neighbors.
knn_classifier_model_object = KNeighborsClassifier()

# n_neighbors is also called as K
# Use of n_neighbors parameter to decide the number of nearest neighbors to be considered.
# knn_classifier_model_object = KNeighborsClassifier(n_neighbors=10)

# Use of p parameter to use Manhattan distance (L1) for computation of nearest neighbors.
# knn_classifier_model_object = KNeighborsClassifier(p=1, metric="minkowski")

# Step-4: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
knn_classifier_model_object.fit(X_train, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = knn_classifier_model_object.predict(X_test)

# b. Predict using single value
predict_90_versicolor = knn_classifier_model_object.predict([[5.5, 2.6, 4.4, 1.2]])     # 90 represents versicolor

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = knn_classifier_model_object.score(X_test, y_test)

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
