import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

df = pd.read_csv("heart.csv")
nb_X = df.drop("target", axis=1)
nb_y = df["target"]

# Train-test split
nb_X_train, nb_X_test, nb_y_train, nb_y_test = train_test_split(nb_X, nb_y, test_size=0.2, random_state=42)

# Train model
nb = GaussianNB()
nb.fit(nb_X_train, nb_y_train)

with open("model/naive-bayes.pkl", "wb") as nb_f:
    pickle.dump(nb, nb_f)

print("Naive Bayes model trained and saved as naive-bayes.pkl")
