import pandas as pandas_package
import numpy as numpy_package
import matplotlib.pyplot as plot_package
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Step-1: Prepare data
# a. Read data from csv file
df_object = pandas_package.read_csv("../../datasets/k-means_clustering/people_income.csv")

# b. Check for any NaN values in dataframe
any_nan_columns = df_object.isna().sum()    # Return all columns with number of NaN values
which_columns_has_nan_values = df_object.columns[df_object.isna().any()]    # Return column names which has NaN values

# c. Prepare x and y axis data from dataframe
x_axis_data = df_object[["Age"]]
y_axis_data = df_object[["Income($)"]]

# d. To view clusters in scatter plot, it is good practice to have x-axis and y-axis data in the range of 0 to 1.
# Create columns for features which are used for x-axis and y-axis where values should be in the range of 0 to 1.
# Use of MinMaxScaler to convert column values to its equivalent value in the range of 0 to 1.
scaler_object = MinMaxScaler()
df_object["age_scaled_0_to_1"] = scaler_object.fit_transform(x_axis_data)
df_object["income_scaled_0_to_1"] = scaler_object.fit_transform(y_axis_data)

# e. Remove any unwanted columns from dataframe which doesn't have impact on prediction value.
df_object.drop(["Name", "Age", "Income($)"], axis="columns", inplace=True)

# f. Prepare data used to predict

# i. Split the available data into train and test data using train_test_split method

# Step-2: Scatter Plot for features of dataset to make sure that clusters can be formed
plot_package.xlabel("Age")
plot_package.ylabel("Salary")
plot_package.title("Salaries w.r.t age")
plot_package.scatter(df_object["age_scaled_0_to_1"], df_object["income_scaled_0_to_1"], marker=".", color="red")
plot_package.show()

# Step-3: Need to identify the number of clusters to be used.
# Value of k represents the number fo clusters
k_values_range = numpy_package.arange(1, 11)
sse_values = []
for i in k_values_range:
      k_means_model_object = KMeans(n_clusters=i)
      y_predicted = k_means_model_object.fit_predict(df_object)
      sse = k_means_model_object.inertia_   # Use of inertia_ to calculate sse for that k value
      sse_values.append(sse)

# Draw elbow plot between k and sse to find out the value of k
plot_package.xlabel("Number of clusters")
plot_package.ylabel("SSE")
plot_package.title("Elbow plot")
plot_package.plot(k_values_range, sse_values)
plot_package.show()

# Step-4: Create K-Means Clustering object
k_means_cluster_model_object = KMeans(n_clusters=3)

# Step-5: Train the model with available data using fit_predict() method
# fit_predict(x_2D_array)
df_object["predicted"] = k_means_cluster_model_object.fit_predict(df_object)

# Step-6: Start predictions and test the model using the predict() method
# predict(x_2D_array)
# a. Predict using test data

# b. Predict using single value


# Step-7: Verify the accuracy of the model using score() method

# Step-8: Get centroid coordinates
get_centroids_coordinates = k_means_cluster_model_object.cluster_centers_   # Returns a 2D array where each value is [x, y]
x_axis_centroids_coordinates = get_centroids_coordinates[:, :1]
y_axis_centroids_coordinates = get_centroids_coordinates[:, 1:2]

# Step-9: Plot scatter plot with clusters and centroid points
plot_package.xlabel("Age")
plot_package.ylabel("Salary")
plot_package.title("Salaries w.r.t age")
cluster_1 = df_object[df_object["predicted"] == 0]
cluster_2 = df_object[df_object["predicted"] == 1]
cluster_3 = df_object[df_object["predicted"] == 2]
plot_package.scatter(cluster_1["age_scaled_0_to_1"], cluster_1["income_scaled_0_to_1"], marker="+", color="pink", label="Cluster-1")
plot_package.scatter(cluster_2["age_scaled_0_to_1"], cluster_2["income_scaled_0_to_1"], marker=".", color="blue", label="Cluster-2")
plot_package.scatter(cluster_3["age_scaled_0_to_1"], cluster_3["income_scaled_0_to_1"], marker="*", color="green", label="Cluster-3")
plot_package.scatter(x_axis_centroids_coordinates, y_axis_centroids_coordinates, marker="*", color="black", label="Centroid")
plot_package.legend()
plot_package.show()

# Step-10: Look at the probability of the predictions using predict_proba() method

# Step-11: Perform other steps like exporting to new csv file, save as pickle.
