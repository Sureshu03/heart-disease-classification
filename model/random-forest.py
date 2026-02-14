import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("heart.csv")
rf_X = df.drop("target", axis=1)
rf_y = df["target"]

# Train-test split
rf_X_train, rf_X_test, rf_y_train, rf_y_test = train_test_split(rf_X, rf_y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(rf_X_train, rf_y_train)

with open("model/random-forest.pkl", "wb") as rf_f:
    pickle.dump(rf, rf_f)

print("Random Forest model trained and saved as random-forest.pkl")
