import joblib as joblib_package

model_object = joblib_package.load("save_model_as_joblib")

predicted_value = model_object.predict([[5000]])
print(predicted_value)
