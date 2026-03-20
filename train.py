import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv("train.csv")

X = df[["age", "salary"]]
y = df["bought"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model train paniyachuu  and saved!")
