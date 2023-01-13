import pandas as pd

DATA_DIRECTORY = 'D:\\Projects\\Drag-flik-RAIS-Hackathon\\data'


label_df = pd.read_excel("data/Totaal data file.xlsx", sheet_name="Ballspeed radar gun ")
label_df = label_df.rename(columns={"Unnamed: 0": "id", "Unnamed: 1": "year"})
label_df = label_df.dropna(axis=0, how="all")
label_df = label_df.replace("err", -1)

data = []

for index, row in label_df.iterrows():
    for i in range(1, len(list(label_df))-1):
        data.append([int(row["id"]), int(row["year"]), i, row[i]])

label_df = pd.DataFrame(data, columns=["id", "year", "sample", "speed"])

mvnx_df = pd.read_pickle('mvnx_data.pkl')
# todo merge the ballspeed.csv based on id, year, and sample
merged = mvnx_df.merge(label_df, on=['year', 'id', 'sample'], how='inner')
print("Labels shape: {} - Features Shape: {} - Inner Join Merged df Shape: {}".format(label_df.shape, mvnx_df.shape, merged.shape))
merged.to_pickle('mvnx_ballspeed_data.pkl')


