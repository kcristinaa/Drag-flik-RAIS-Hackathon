import glob
import os
import pandas as pd
from tqdm import tqdm
from load_mvnx import load_mvnx
from utility_functions import read_single_mvnx_to_df, read_mvnx_metadata

DATA_DIRECTORY = 'D:\\Projects\\Drag-flik-RAIS-Hackathon\\data'

file_names = []
for f in glob.glob(os.path.join(DATA_DIRECTORY, '**\\*.mvnx'), recursive=True):
    file_names.append(f)
print("Identified {} .mvnx files...".format(len(file_names)))

df = pd.DataFrame()
for file_name in tqdm(file_names):
    mvnx_file = load_mvnx(os.path.join(DATA_DIRECTORY, file_name))
    if mvnx_file is None:
        print("Not possible to parse file {}".format(file_name))
        continue
    file_df = read_single_mvnx_to_df(mvnx_file, disable_print=True)
    # add metadata
    year, id, sample, gender = read_mvnx_metadata(mvnx_file, file_name)
    if 'A' in year or 'B' in year:
        year = '2021'

    file_df = pd.concat([file_df, pd.Series(year, name='year'), pd.Series(id, name='id'), pd.Series(sample, name='sample'), pd.Series(gender, name='gender')], axis=1)
    file_df[['year', 'gender', 'id', 'sample']] = file_df[['year', 'gender', 'id', 'sample']].ffill()
    df = pd.concat([df, file_df], axis=0)

df = df.astype({'year': int, 'id': int, 'sample': int})
df.to_pickle('mvnx_data.pkl')

