import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("heart.csv")
log_X = df.drop("target", axis=1)
log_y = df["target"]

# Train-test split
log_X_train, log_X_test, log_y_train, log_y_test = train_test_split(log_X, log_y, test_size=0.2, random_state=42)

# Train model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(log_X_train, log_y_train)

with open("model/logistic-regression.pkl", "wb") as log_f:
    pickle.dump(log_reg, log_f)

print("Logistic Regression model trained and saved as logistic-regression.pkl")


