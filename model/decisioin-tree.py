import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("heart.csv")
dt_X = df.drop("target", axis=1)
dt_y = df["target"]

# Train-test split
dt_X_train, dt_X_test, dt_y_train, dt_y_test = train_test_split(dt_X, dt_y, test_size=0.2, random_state=42)

# Train model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(dt_X_train, dt_y_train)

with open("model/decision-tree.pkl", "wb") as dt_f:
    pickle.dump(dt, dt_f)

print("Decision Tree model trained and saved as decision-tree.pkl")



