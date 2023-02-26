import pandas as pandas_package
import matplotlib.pyplot as plot_package
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as seaborn_package

# Step-1: Prepare data
# a. Read data from csv file
df_object = pandas_package.read_csv("../../datasets/decision_tree_classifier/employees_salaries.csv")

# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()

# c. Prepare x and y axis data from dataframe
x_axis_data = df_object.drop("salary_more_then_100k", axis="columns")
y_axis_data = df_object["salary_more_then_100k"]

# d. Since x_axis_data contains text values which can't be understood by ML model,
# converting text values to its equivalent numeric values using LabelEncoder()
company_label_encoder = LabelEncoder()
x_axis_data["company_numeric"] = company_label_encoder.fit_transform(x_axis_data["company"])
job_label_encoder = LabelEncoder()
x_axis_data["job_numeric"] = job_label_encoder.fit_transform(x_axis_data["job"])
degree_label_encoder = LabelEncoder()
x_axis_data["degree_numeric"] = degree_label_encoder.fit_transform(x_axis_data["degree"])

# e. Prepare final x_axis_data with numeric values instead of labels
# 0 stands for company, 1 stands for job and 2 stands for degree
x_axis_numeric_data = x_axis_data.drop(["company", "job", "degree"], axis="columns")

# f. Prepare data used to predict

# g. Split the available data into train and test data using train_test_split method
# 20% of available dataset to be used for test data
# 80% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_numeric_data, y_axis_data, test_size=0.2)

# Step-2: Create decision tree classifier object
decision_tree_classifier_model_object = DecisionTreeClassifier()

# Step-3: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
decision_tree_classifier_model_object.fit(X_train, y_train)

# Step-4: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = decision_tree_classifier_model_object.predict(X_test)

# b. Predict using single value
predict_google_programmer_bachelors_salary = decision_tree_classifier_model_object.predict([[2, 1, 0]])     # 0 or 1
predict_facebook_executive_masters_salary = decision_tree_classifier_model_object.predict([[1, 2, 1]])     # 1

# Step-5: Verify the accuracy of the model using score() method
model_accuracy = decision_tree_classifier_model_object.score(X_test, y_test)

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
