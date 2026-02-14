import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

df = pd.read_csv("heart.csv")
xg_X = df.drop("target", axis=1)
xg_y = df["target"]

# Train-test split
xg_X_train, xg_X_test, xg_y_train, xg_y_test = train_test_split(xg_X, xg_y, test_size=0.2, random_state=42)

# Train model
xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(xg_X_train, xg_y_train)

with open("model/XGboost.pkl", "wb") as xg_f:
    pickle.dump(xgb_model, xg_f)

print("Random Forest model trained and saved as XGBoost.pkl")
