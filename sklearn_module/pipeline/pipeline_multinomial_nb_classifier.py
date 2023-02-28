import pandas as pandas_package
from sklearn.preprocessing import LabelEncoder
# For text content
from sklearn.feature_extraction.text import CountVectorizer
# Multinomial Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
# Pipeline
from sklearn.pipeline import Pipeline

# Step-1: Prepare data
# a. Read data from csv file
df_object = pandas_package.read_csv("../../datasets/naive_bayes/multinomial_naive_bayes/spam_mail_detection.csv")

# b. Create a column with numeric values represents categories
category_label_encoder = LabelEncoder()
df_object["spam"] = category_label_encoder.fit_transform(df_object["Category"])

# c. Prepare x and y axis data from dataframe
x_axis_data = df_object["Message"]
y_axis_data = df_object["spam"]

# d. Prepare data used to predict
sample_emails = [
    "Hey, Welcome to python",
    "Free! Up to 100% discount on product"
]

# e. Split the available data into train and test data using train_test_split method
# 25% of available dataset to be used for test data
# 75% of available dataset to be used for train data
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.25)

# Step-2: Create a pipeline object with Count Vectorizer and Multinomial Naive Bayes Classifier
multinomial_nb_classifier_pipeline = Pipeline(steps=[
    ("vectorizer", CountVectorizer()),
    ("model", MultinomialNB())
])

# Step-3: Train the model with train data using fit() method
# fit(x_2D_array, y_1D_or_2D_array)
multinomial_nb_classifier_pipeline.fit(X_train, y_train)

# Step-5: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data
predict_test_data = multinomial_nb_classifier_pipeline.predict(X_test)

# b. Predict using sample emails
predict_0_1 = multinomial_nb_classifier_pipeline.predict(sample_emails)     # [0, 1]

# Step-6: Verify the accuracy of the model using score() method
model_accuracy = multinomial_nb_classifier_pipeline.score(X_test, y_test)

# Step-7: Look at the probability of the predictions using predict_proba() method

# Step-8: Perform other steps like exporting to new csv file, save as pickle.
