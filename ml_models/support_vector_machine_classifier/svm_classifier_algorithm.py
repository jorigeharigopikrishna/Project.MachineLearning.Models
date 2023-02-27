import pandas as pandas_package
import matplotlib.pyplot as plot_package
import seaborn as seaborn_package
# SVM classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# Using built-in iris flower dataset
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

# Step-1: Prepare data
# a. Read data from built-in dataset of iris flower
iris_dataset = load_iris()   # use of iris flower dataset from sklearn datasets
df_object = pandas_package.DataFrame(iris_dataset["data"], columns=iris_dataset["feature_names"])
df_object["target"] = iris_dataset["target"]
# Create a flower_name column by using apply() method using target_names column
df_object["flower_name"] = df_object.target.apply(lambda x: iris_dataset.target_names[x])  # This is labelled column which can't be used in ML

# b. Prepare dataframes for each version of iris flower
# Setosa version of Iris flower is from 0-50 in iris dataset
setosa_iris_df_object = df_object[df_object.target == 0]
# Versicolor version of Iris flower is from 51-100 in iris dataset
versicolor_iris_df_object = df_object[df_object.target == 1]
# Virginica version of Iris flower is from 101-150 in iris dataset
virginica_iris_df_object = df_object[df_object.target == 2]

# c. Check for any NaN values in dataframe
setosa_any_nan_columns = setosa_iris_df_object.isna().sum()
versiocolor_any_nan_columns = versicolor_iris_df_object.isna().sum()
virginica_any_nan_columns = virginica_iris_df_object.isna().sum()

# d. Prepare x and y axis data from dataframe
x_axis_data = df_object.drop(["target", "flower_name"], axis="columns")
y_axis_data = df_object["target"]

# e. Prepare data used to predict

# f. Split the available data into train and test data using train_test_split method
# 20% of available dataset to be used for test data
# 80% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.2)

# Step-2: Plot scatter plots to make sure that categories can be separated by decision boundary
# to use SVM Classifier
# Scatter Plots between sepal length and sepal width for all versions of Iris flower
plot_package.xlabel("Sepal length (cm)")
plot_package.ylabel("Sepal width (cm)")
plot_package.title("Types of Iris Flowers")
plot_package.scatter(setosa_iris_df_object["sepal length (cm)"], setosa_iris_df_object["sepal width (cm)"], marker="+", color="blue", label="Setosa iris")
plot_package.scatter(versicolor_iris_df_object["sepal length (cm)"], versicolor_iris_df_object["sepal width (cm)"], marker="*", color="green", label="Versicolor iris")
plot_package.legend()
plot_package.show()

# Scatter Plots between petal length and petal width for all versions of Iris flower
plot_package.xlabel("Petal length (cm)")
plot_package.ylabel("Petal width (cm)")
plot_package.title("Types of Iris Flowers")
plot_package.scatter(setosa_iris_df_object["petal length (cm)"], setosa_iris_df_object["petal width (cm)"], marker="+", color="blue", label="Setosa iris")
plot_package.scatter(versicolor_iris_df_object["petal length (cm)"], versicolor_iris_df_object["petal width (cm)"], marker="*", color="green", label="Versicolor iris")
plot_package.legend()
plot_package.show()

# Step-3: Create support vector machine (svm) classifier object
# Using default values of C=1, gamma="scale" and kernel="rbf"
svm_classifier_model_object = SVC()

# Use of Regularization (C) parameter to control the regularization.
# svm_classifier_model_object = SVC(C=10)

# Use of Gamma parameter to control the gamma.
# svm_classifier_model_object = SVC(gamma=10)

# Use of Kernel parameter to perform any transformation.
# svm_classifier_model_object = SVC(kernel="linear")

# Step-4: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
svm_classifier_model_object.fit(X_train, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = svm_classifier_model_object.predict(X_test)

# b. Predict using single value
predict_90_versicolor = svm_classifier_model_object.predict([[5.5, 2.6, 4.4, 1.2]])     # 90 represents versicolor

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = svm_classifier_model_object.score(X_test, y_test)

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
