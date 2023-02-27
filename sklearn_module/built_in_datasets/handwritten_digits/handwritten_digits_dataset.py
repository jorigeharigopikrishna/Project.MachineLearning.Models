import pandas as pandas_package
# Using built-in handwritten digits dataset
from sklearn.datasets import load_digits
import matplotlib.pyplot as plot_package

# Handwritten digits will be from 0 to 9
# Each handwritten digit will be of 8*8 matrix
# There are a total of 1797 handwritten images as part of this built-in dataset.
digits_dataset = load_digits()   # use of handwritten digits dataset from sklearn datasets
print(dir(digits_dataset))
# Handwritten digits dataset has following data
# ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']

# Since each digit is of 8*8 matrix the columns will be 64
# feature_names - ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']
# target_names - 0 to 9 i.e., [0 1 2 3 4 5 6 7 8 9]
# data - Values for each cell in 8*8 matrix, so there will be 64 cells representing each handwritten digit.
# images - Original images of handwritten digits
# target - Original numerical value of handwritten digit

df_object = pandas_package.DataFrame(digits_dataset["data"], columns=digits_dataset["feature_names"])
df_object["target"] = digits_dataset["target"]

# Hand written digit of 2
plot_package.gray()
plot_package.matshow(digits_dataset.images[2])
plot_package.show()
