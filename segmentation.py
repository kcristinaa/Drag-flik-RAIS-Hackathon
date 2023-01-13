import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feats, h_feats[0]),
            nn.ReLU(),
            nn.Linear(h_feats[0], out_feats),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def plot_number_of_segments(df):
    analytics = df.groupby(["year", "id", "sample"])
    labels = []
    values = []
    for name, group in analytics:
        labels.append(f"{name[0]}-{name[1]}-{name[2]}")
        values.append(group["segment"].count())
    y_pos = range(len(labels))
    plt.figure(figsize=(150, 25))
    plt.bar(y_pos, values)
    plt.xticks(y_pos, labels, rotation=90)
    plt.xlabel("Year-Id-Trial")
    plt.ylabel("Num. of Segments")
    plt.savefig("segments_count.png")


def plot_heatmap_normalized(y_test, pred, class_names):
    cm = confusion_matrix(y_test, pred)
    conf_matrix_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
    df_cm_norm = pd.DataFrame(conf_matrix_norm, index=class_names, columns=class_names)
    sns.heatmap(df_cm_norm, annot=True)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("HeatMap - Normalized")
    plt.savefig("heatmap_normalized.png")


if __name__ == "__main__":
    data = pd.read_pickle("data/mvnx_ballspeed_segmentation.pkl")
    data = data.drop(
        ['gender', 'LeftToe_acc_0', 'LeftToe_acc_1', 'LeftToe_acc_2', 'LeftToe_angular_acc_0', 'LeftToe_angular_acc_1',
         'LeftToe_angular_acc_2', 'LeftToe_vel_0', 'LeftToe_vel_1', 'LeftToe_vel_2', 'LeftToe_angular_vel_0',
         'LeftToe_angular_vel_1', 'LeftToe_angular_vel_2', 'LeftToe_ori_0', 'LeftToe_ori_1', 'LeftToe_ori_2',
         'LeftToe_ori_3', 'LeftToe_pos_0', 'LeftToe_pos_1', 'LeftToe_pos_2'], axis=1)

    class_names = ["T2", "T3", "T4", "T5"]
    segments = data.loc[data["segment"].isin(class_names)]
    plot_number_of_segments(segments)

    le = LabelEncoder()
    le.fit(segments["segment"])
    integer_encoded = le.transform(segments["segment"])

    # "time", "year", "id", "sample", "speed", "key", "segment"
    X_train, X_test, y_train, y_test = train_test_split(
        segments[segments.columns.difference(["year", "id", "sample", "speed", "key", "segment"])],
        OneHotEncoder(sparse=False).fit_transform(integer_encoded.reshape(len(integer_encoded), 1)),
        random_state=0,
        shuffle=False,
        test_size=0.1
    )

    # clf = RandomForestClassifier(n_jobs=-1)
    # clf.fit(X_train, y_train)
    # pred = clf.predict(X_test)
    # score = f1_score(y_test, pred, average="micro")
    # print(f"F1: {score}")

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(device)

    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float).to(device)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float).to(device)
    y_train = torch.tensor(y_train).to(device)
    y_test = np.argmax(y_test, axis=1)

    net = MLP(X_train.size(dim=1), [30], len(le.classes_)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    for epoch in tqdm(range(0, 5000)):
        current_loss = 0.0
        optimizer.zero_grad()
        out = net(X_train)
        loss = loss_function(out, y_train)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()

    pred = np.argmax(net(X_test).to("cpu").detach().numpy(), axis=1)
    score = f1_score(y_test, pred, average="micro")
    print(f"F1: {score}")

    plot_heatmap_normalized(y_test, pred, class_names)
