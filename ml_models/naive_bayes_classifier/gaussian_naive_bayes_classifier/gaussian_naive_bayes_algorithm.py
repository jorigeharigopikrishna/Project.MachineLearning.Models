import pandas as pandas_package
import seaborn as seaborn_package
import matplotlib.pyplot as plot_package
from sklearn.preprocessing import LabelEncoder
# Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Step-1: Prepare data
# a. Read data from csv file
df_object = pandas_package.read_csv("../../../datasets/naive_bayes/gaussian_naive_bayes/titanic_crash_survival_dataset.csv")

# b. Remove any unwanted columns from dataframe which doesn't have impact on prediction value.
df_object.drop(["PassengerId", "Name", "SibSp", "Parch", "Embarked", "Cabin", "Ticket"], axis="columns", inplace=True)

# c. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()    # Return all columns with number of NaN values
which_columns_has_nan_values = df_object.columns[df_object.isna().any()]    # Return column names which has NaN values

# d. Replace nan values with its previous value
df_object["Age"].fillna(method="ffill", inplace=True)

# e. Replace categories with its numeric values
sex_label_encoder = LabelEncoder()
df_object["male"] = sex_label_encoder.fit_transform(df_object["Sex"])

# f. Remove category columns from dataframe as ML model wont work with labels
df_object.drop("Sex", axis="columns", inplace=True)

# g. Prepare x and y axis data from dataframe
x_axis_data = df_object.drop("Survived", axis="columns")
y_axis_data = df_object["Survived"]

# h. Prepare data used to predict

# i. Split the available data into train and test data using train_test_split method
# 20% of available dataset to be used for test data
# 80% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.2)

# Step-2: Draw a histogram to make sure that there will be a normal distribution or bell curve to use gaussian naive bayes classifier
plot_package.xlabel("Age")
plot_package.ylabel("Frequency")
seaborn_package.histplot(df_object["Age"], kde=True)
plot_package.show()

# Step-3: Create Gaussian Naive Bayes Classifier object
# Using conditional probability - P(Survived/Sex&Class&Age&Fare)
gaussian_nb_classifier_model_object = GaussianNB()

# Step-4: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
gaussian_nb_classifier_model_object.fit(X_train, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = gaussian_nb_classifier_model_object.predict(X_test)

# b. Predict using single value
predict_1_62_80_female = gaussian_nb_classifier_model_object.predict([[1, 62, 80, 0]])     # 829 predicts 1

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = gaussian_nb_classifier_model_object.score(X_test, y_test)

# Step-7: Look at the probability of the predictions using predict_proba() method

# Step-8: Perform other steps like exporting to new csv file, save as pickle.
