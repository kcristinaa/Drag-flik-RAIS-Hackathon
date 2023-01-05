import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

data = pd.read_pickle("mvnx_merged_data_train_validation_test_preprocessed_har.pkl")

segments = data.loc[data["segment"].isin(["T2", "T3", "T4", "T5"])]

# print(list(segments.columns))

X_train, X_test, y_train, y_test = train_test_split(
    segments[segments.columns.difference(["time", "year", "id", "sample", "speed", "key", "segment"])],
    segments["segment"],
    random_state=0
)

# clf = Perceptron(n_jobs=-1)
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = f1_score(y_test, pred, average="micro")

print(f"F1: {score}")

