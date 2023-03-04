import pandas as pandas_package
# Using built-in handwritten digits dataset
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Prepare data
digits_dataset = load_digits()

df_object = pandas_package.DataFrame(digits_dataset["data"], columns=digits_dataset["feature_names"])
df_object["target"] = digits_dataset["target"]

# Original x and y axis data
x_axis_data = df_object.drop("target", axis="columns")
y_axis_data = df_object["target"]

# Scaled x and y axis data
standard_scaler = StandardScaler()
scaled_x_axis_data = standard_scaler.fit_transform(x_axis_data)
scaled_y_axis_data = df_object["target"]
# Split original x and y axis dataset
X_train, X_test, y_train, y_test = train_test_split(x_axis_data, y_axis_data, test_size=0.2, random_state=10)
# Split scaled x and y axis dataset
X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test = train_test_split(scaled_x_axis_data, scaled_y_axis_data, test_size=0.2, random_state=10)

# LogisticRegression using original x and y axis data
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_model_score = lr_model.score(X_test, y_test)

# LogisticRegression using scaled x and y axis data
lr_model_scaled = LogisticRegression()
lr_model_scaled.fit(X_scaled_train, y_scaled_train)
lr_model_scaled_score = lr_model_scaled.score(X_scaled_test, y_scaled_test)
print(f"Original: Without scaling score {lr_model_score} and With scaling score {lr_model_scaled_score}")

# PCA dataset with original x axis data
pca_object = PCA(0.95)  # Extract 95% of useful information from original features
pca_x_axis_dataset = pca_object.fit_transform(x_axis_data)
pca_df_object = pandas_package.DataFrame(pca_x_axis_dataset)
pca_object_columns = len(pca_df_object.columns)
# PCA dataset with scaled x axis data
pca_scaled_object = PCA(0.95)   # Extract 95% of useful information from original features
pca_scaled_x_axis_dataset = pca_scaled_object.fit_transform(scaled_x_axis_data)
pca_scaled_df_object = pandas_package.DataFrame(pca_scaled_x_axis_dataset)
pca_scaled_object_columns = len(pca_scaled_df_object.columns)
# Split PCA original x and y axis dataset
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(pca_x_axis_dataset, y_axis_data, test_size=0.2, random_state=10)
# Split PCA scaled x and y axis dataset
X_scaled_pca_train, X_scaled_pca_test, y_scaled_pca_train, y_scaled_pca_test = train_test_split(pca_scaled_x_axis_dataset, scaled_y_axis_data, test_size=0.2, random_state=10)

# LogisticRegression using PCA original x and y axis data
lr_model_pca = LogisticRegression()
lr_model_pca.fit(X_pca_train, y_pca_train)
lr_model_pca_score = lr_model_pca.score(X_pca_test, y_pca_test)
# LogisticRegression using PCA Scaled x and y axis data
lr_model_pca_scaled = LogisticRegression()
lr_model_pca_scaled.fit(X_scaled_pca_train, y_scaled_pca_train)
lr_model_pca_scaled_score = lr_model_pca_scaled.score(X_scaled_pca_test, y_scaled_pca_test)
print(f"PCA: Without scaling columns {pca_object_columns} and With scaling columns {pca_scaled_object_columns}")
print(f"PCA: Without scaling score {lr_model_pca_score} and With scaling score {lr_model_pca_scaled_score}")
