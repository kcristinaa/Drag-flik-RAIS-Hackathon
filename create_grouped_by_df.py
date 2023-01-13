import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from dataprep.eda import create_report
from pycaret.regression import *
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle('data/mvnx_ballspeed_segmentation.pkl')
df = df[~df['speed'].isnull()]
df = df.drop(df[df.speed == -1].index)
cols = list(df.columns)
columns_excluded = df[['time', 'year', 'id', 'sample', 'speed', 'key', 'segment']]
for col in columns_excluded:
    cols.remove(col)

dff = pd.DataFrame()

for col in cols:
    df_new = df.groupby(['key', 'speed']).aggregate({col: ['mean', 'std']})
    dff = pd.concat([dff, df_new], axis=1)

dff.reset_index()
dff.head()

dff.columns = list(map(''.join, dff.columns.values))
dff = dff.reset_index()

dff.to_pickle('data/mvnx_ballspeed_segmentation_grouped_by.pkl')
