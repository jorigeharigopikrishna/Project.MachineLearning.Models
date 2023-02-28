import pandas as pandas_package
from sklearn.preprocessing import LabelEncoder
# For text content
from sklearn.feature_extraction.text import CountVectorizer
# Multinomial Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Step-1: Prepare data
# a. Read data from csv file
df_object = pandas_package.read_csv("../../../datasets/naive_bayes/multinomial_naive_bayes/spam_mail_detection.csv")
df_object_groups_stats = df_object.groupby("Category").describe()

# b. Remove any unwanted columns from dataframe which doesn't have impact on prediction value.

# c. Check for any NaN values in dataframe
# any_nan_columns = df_object.isna().sum()    # Return all columns with number of NaN values
# which_columns_has_nan_values = df_object.columns[df_object.isna().any()]    # Return column names which has NaN values

# e. Create a column with numeric values represents categories
category_label_encoder = LabelEncoder()
df_object["spam"] = category_label_encoder.fit_transform(df_object["Category"])

# f. Remove category columns from dataframe as ML model wont work with labels
df_object.drop("Category", axis="columns", inplace=True)

# g. Prepare x and y axis data from dataframe
x_axis_data = df_object["Message"]
y_axis_data = df_object["spam"]

# h. Prepare data used to predict
sample_emails = [
    "Hey, Welcome to python",
    "Free! Up to 100% discount on product"
]

# i. Split the available data into train and test data using train_test_split method
# 25% of available dataset to be used for test data
# 75% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.25)

# Step-2: Use count vectorizer
count_vectorizer = CountVectorizer()
x_train_sparse_matrix = count_vectorizer.fit_transform(X_train) # Use of fit_transform() for train data
x_test_sparse_matrix = count_vectorizer.transform(X_test)   # Use of transform() for test data

# Step-3: Create Multinomial Naive Bayes Classifier object
multinomial_nb_classifier_model_object = MultinomialNB()

# Step-4: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
multinomial_nb_classifier_model_object.fit(x_train_sparse_matrix, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = multinomial_nb_classifier_model_object.predict(x_test_sparse_matrix)

# b. Predict using sample emails
emails_sparse_matrix = count_vectorizer.transform(sample_emails)
predict_0_1 = multinomial_nb_classifier_model_object.predict(emails_sparse_matrix)     # [0, 1]

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = multinomial_nb_classifier_model_object.score(x_test_sparse_matrix, y_test)

# Step-7: Look at the probability of the predictions using predict_proba() method

# Step-8: Perform other steps like exporting to new csv file, save as pickle.
