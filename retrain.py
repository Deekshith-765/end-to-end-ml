import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Check drift
with open("drift_status.txt", "r") as f:
    drift = f.read() == "True"

if drift:
    print("Retraining modell...")

    old_data = pd.read_csv("train.csv")
    new_data = pd.read_csv("newdata.csv")

    
    new_data["bought"] = [0, 1, 0]

    full_data = pd.concat([old_data, new_data])

    X = full_data[["age", "salary"]]
    y = full_data["bought"]

    print("Unique classes:", set(y))

    
    if len(set(y)) < 2:
        print("Not enough classes. Skipping retraining.")
        exit()

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "model/model.pkl")

    print("Model retrain paniaachuu !")

else:
    print("Retrain pana vendampaaa")
