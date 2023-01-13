import pandas as pd
import numpy as np
import glob
import os

from tqdm import tqdm
from dataprep.eda import create_report

from utility_functions import get_year_from_filename

file_names = []
for f in glob.glob(os.path.join("data", "keyframes", '**\\*.csv'), recursive=True):
    file_names.append(f)

for f in glob.glob(os.path.join("data", "keyframes", '**\\*.xlsx'), recursive=True):
    file_names.append(f)

print("Identified {} .csv and .xlsx files...".format(len(file_names)))
print(file_names)

segmentation_df = pd.DataFrame(columns=['bodyNumber', 'trial', 'year', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'BR', 'BV', 'ORWF'])
# BEFORE RUNNING: Go to file data\keyframes\2021_A\Keyframes\Body69_keyframes.csv and delete the final comma in line 15
for file_name in tqdm(file_names):
    year = get_year_from_filename(file_name)
    if str.endswith(file_name, 'csv'):
        temp = pd.read_csv(file_name, sep=';|,', engine='python')
    else:
        temp = pd.read_excel(file_name)
    # add year
    temp.loc[:, 'year'] = year
    # remove empty, unnamed columns
    temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
    # remove filename column present only in 2021_B folder
    temp = temp.loc[:, ~temp.columns.str.contains('filename')]
    # remove white spaces from column names
    temp.columns = temp.columns.str.replace(' ', '')
    # append temp to main df
    segmentation_df = pd.concat([segmentation_df, temp], axis=0)
    if len(segmentation_df.columns) != 13:
        print("There is a problem with the formatting of file {}. Please correct it before proceeding.".format(file_name))
        print(segmentation_df.tail())
        break

print("*********** Dataframe Content: ***********\n")
print(segmentation_df.head())

print("*********** Dataframe Description: ***********\n")
print(segmentation_df.describe())

print("*********** Dataframe Null Values per Column: ***********\n")
print(segmentation_df.isna().sum())

print("Reading mvnx data...")
# mvnx = pd.read_pickle(os.path.join("data", "mvnx_merged_data_train_validation_test_preprocessed.pkl"))
mvnx_ballspeed_df = pd.read_pickle(os.path.join("data", "mvnx_ballspeed_data.pkl"))
mvnx_ballspeed_df = mvnx_ballspeed_df.drop_duplicates()
# mvnx.time = mvnx.time.astype(int)
print("Creating key...")
mvnx_ballspeed_df["key"] = mvnx_ballspeed_df['year'].astype(str) + '_' + mvnx_ballspeed_df['id'].astype(str) + '_' + mvnx_ballspeed_df['sample'].astype(str)
print(mvnx_ballspeed_df.head())

print(mvnx_ballspeed_df.time.isna().sum())

mvnx_ballspeed_df = mvnx_ballspeed_df[mvnx_ballspeed_df.time.notna()]
mvnx_ballspeed_df.time = mvnx_ballspeed_df.time.astype(int)
print(mvnx_ballspeed_df.head())

segmentation_df["key"] = segmentation_df['year'].astype(str) + '_' + segmentation_df['bodyNumber'].astype(str) + '_' + segmentation_df['trial'].astype(str)
print(segmentation_df.head())

mvnx_ballspeed_df[['segment']] = np.NaN
print(mvnx_ballspeed_df)

for row in tqdm(segmentation_df.iterrows()):
    # get data from keyframes row
    key = row[1].key
    T2 = row[1].T2
    T3 = row[1].T3
    T4 = row[1].T4
    T5 = row[1].T5
    T6 = row[1].T6
    # adding True (1) to the rows referring to each time segment
    mvnx_ballspeed_df.loc[(mvnx_ballspeed_df["key"]==key) & (mvnx_ballspeed_df["time"] >= T2) & (mvnx_ballspeed_df["time"] < T3), "segment"] = "T2"
    mvnx_ballspeed_df.loc[(mvnx_ballspeed_df["key"]==key) & (mvnx_ballspeed_df["time"] >= T3) & (mvnx_ballspeed_df["time"] < T4), "segment"] = "T3"
    mvnx_ballspeed_df.loc[(mvnx_ballspeed_df["key"]==key) & (mvnx_ballspeed_df["time"] >= T4) & (mvnx_ballspeed_df["time"] < T5), "segment"] = "T4"
    mvnx_ballspeed_df.loc[(mvnx_ballspeed_df["key"]==key) & (mvnx_ballspeed_df["time"] >= T5) & (mvnx_ballspeed_df["time"] < T6), "segment"] = "T5"

print(mvnx_ballspeed_df.head())
print(mvnx_ballspeed_df.loc[mvnx_ballspeed_df.key=="2017_10_1"])

mvnx_ballspeed_segmentation_df = mvnx_ballspeed_df.loc[mvnx_ballspeed_df.segment.notnull()]
print("Original shape: {} - HAR shape: {}".format(mvnx_ballspeed_df.shape[0], mvnx_ballspeed_segmentation_df.shape[0]))
print(mvnx_ballspeed_segmentation_df.head())

mvnx_ballspeed_segmentation_df.to_pickle(os.path.join("data", "mvnx_ballspeed_segmentation.pkl"))

