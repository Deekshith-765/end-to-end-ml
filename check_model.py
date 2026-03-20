import joblib

model = joblib.load("model/model.pkl")

print("Model:", model)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
