import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv("heart.csv")
knn_X = df.drop("target", axis=1)
knn_y = df["target"]

# Train-test split
knn_X_train, knn_X_test, knn_y_train, knn_y_test = train_test_split(knn_X, knn_y, test_size=0.2, random_state=42)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(knn_X_train, knn_y_train)

with open("model/knn.pkl", "wb") as knn_f:

    pickle.dump(knn, knn_f)

print("Nearest neighbour model trained and saved as knn.pkl")

