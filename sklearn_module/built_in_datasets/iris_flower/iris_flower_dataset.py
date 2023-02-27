import pandas as pandas_package
# Using built-in iris flower dataset
from sklearn.datasets import load_iris
import matplotlib.pyplot as plot_package

# Iris flower has three variations - setosa, versiocolor, virginica
# These three variations will be based on sepal length, sepal width, petal length and petal width.
iris_dataset = load_iris()   # use of iris flower dataset from sklearn datasets

# Iris dataset has following data
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

# feature_names - ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# target_names - Three types of flowers - ['setosa' 'versicolor' 'virginica']
# data - Values for sepal length, sepal width, petal length and petal width.
# target - Equivalent numerical values representing target_names where setosa = 0, versicolor = 1, virginica = 2

df_object = pandas_package.DataFrame(iris_dataset["data"], columns=iris_dataset["feature_names"])
df_object["target"] = iris_dataset["target"]
df_object["flower_name"] = df_object.target.apply(lambda x: iris_dataset.target_names[x]) # This is labelled column which can't be used in ML

# Setosa version of Iris flower is from 0-50 in iris dataset
setosa_iris_df_object = df_object[df_object.target == 0]

# Versicolor version of Iris flower is from 51-100 in iris dataset
versicolor_iris_df_object = df_object[df_object.target == 1]

# Virginica version of Iris flower is from 101-150 in iris dataset
virginica_iris_df_object = df_object[df_object.target == 2]

# Scatter Plots between sepal length and sepal width for all versions of Iris flower
plot_package.xlabel("Sepal length (cm)")
plot_package.ylabel("Sepal width (cm)")
plot_package.title("Types of Iris Flowers")
plot_package.scatter(setosa_iris_df_object["sepal length (cm)"], setosa_iris_df_object["sepal width (cm)"], marker="+", color="blue", label="Setosa iris")
plot_package.scatter(versicolor_iris_df_object["sepal length (cm)"], versicolor_iris_df_object["sepal width (cm)"], marker="*", color="green", label="Versicolor iris")
plot_package.scatter(virginica_iris_df_object["sepal length (cm)"], virginica_iris_df_object["sepal width (cm)"], marker=".", color="red", label="Virginica iris")
plot_package.legend()
plot_package.show()

# Scatter Plots between petal length and petal width for all versions of Iris flower
plot_package.xlabel("Petal length (cm)")
plot_package.ylabel("Petal width (cm)")
plot_package.title("Types of Iris Flowers")
plot_package.scatter(setosa_iris_df_object["petal length (cm)"], setosa_iris_df_object["petal width (cm)"], marker="+", color="blue", label="Setosa iris")
plot_package.scatter(versicolor_iris_df_object["petal length (cm)"], versicolor_iris_df_object["petal width (cm)"], marker="*", color="green", label="Versicolor iris")
plot_package.scatter(virginica_iris_df_object["petal length (cm)"], virginica_iris_df_object["petal width (cm)"], marker=".", color="red", label="Virginica iris")
plot_package.legend()
plot_package.show()
