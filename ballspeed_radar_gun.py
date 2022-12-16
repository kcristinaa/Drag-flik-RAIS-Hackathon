import pandas as pd


df = pd.read_excel("data/Totaal data file.xlsx", sheet_name="Ballspeed radar gun ")
df = df.rename(columns={"Unnamed: 0": "id", "Unnamed: 1": "year"})
df = df.dropna(axis=0, how="all")
df = df.replace("err", -1)

data = []

for index, row in df.iterrows():
    for i in range(1, len(list(df))-1):
        data.append([int(row["id"]), int(row["year"]), i, row[i]])

df = pd.DataFrame(data, columns=["id", "year", "sample", "speed"])
df.to_csv("data/ballspeed.csv", index=False)
