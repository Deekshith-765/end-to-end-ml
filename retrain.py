import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

with open("drift_status.txt", "r") as f:
    content = f.read()
    drift = "data_drift=True" in content or "True" in content



if drift:
    print("Drift detected retraining..........")

    old_data = pd.read_csv("train.csv")
    new_data = pd.read_csv("newdata.csv")


    new_data["bought"] = [0, 1, 0]

    full_data = pd.concat([old_data, new_data])

    X = full_data[["age", "salary"]]
    y = full_data["bought"]

    if len(set(y)) < 2:
        print(" Not enough classes. retraining theva ila .")
        exit()


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        old_model = joblib.load("model/model.pkl")
        old_score = old_model.score(X_test, y_test)
    except:
        print(" No old model found → training fresh")
        old_model = None
        old_score = 0

    print(f"Old Model Accuracy: {old_score}")


    new_model = LogisticRegression()
    new_model.fit(X_train, y_train)

    new_score = new_model.score(X_test, y_test)
    print(f"New Model Accuracy: {new_score}")

    if new_score > old_score:
        joblib.dump(new_model, "model/model.pkl")
        print(" New model deployed (better performance)")
    else:
        print("Old model retained (new one worse)")

else:
    print(" No drift → No retraining needed")