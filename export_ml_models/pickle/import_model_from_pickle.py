import pickle as python_pickle_module

with open("save_model_as_pickle", "rb") as file:    # rb stands for read binary data
    imported_ml_model = python_pickle_module.load(file) # load() used to load the model

predicted_value = imported_ml_model.predict([[5000]])
print(predicted_value)
