import pandas as pd

# Load datasets
old_data = pd.read_csv("train.csv")
new_data = pd.read_csv("newdata.csv")

# Compare mean values
old_mean = old_data[["age", "salary"]].mean()
new_mean = new_data.mean()

difference = abs(old_mean - new_mean)

print("Difference:\n", difference)

# Threshold
threshold = 10000

if difference["salary"] > threshold:
    print("Drift kandupidichachuuuuu!")
    drift = True
else:
    print("Drift ila paaa ")
    drift = False

# Save result
with open("drift_status.txt", "w") as f:
    f.write(str(drift))
