# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
import xgboost 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#time the code 
start = time.time()

#################################################################################################
# Eva's dataframe - merged data + 5 extra features  

# READ THE pcl FILE
df_merged_added_features = pd.read_pickle(
    'C:/Users/nickg/Desktop/Hackathon/final_without_NaNs_df.pkl')

#Basic df elements
# print(df_merged_added_features.head(10))
# print(df_merged_added_features.shape)
# print(df_merged_added_features.columns)

#speed to index -1 
df_speed = df_merged_added_features['speed']
df_merged_added_features = df_merged_added_features.drop(['speed'], axis = 1)
df_merged_added_features['speed'] = df_speed

#drop key
df_merged_added_features = df_merged_added_features.drop(['key'], axis = 1)
print(df_merged_added_features.head(10))
print(df_merged_added_features.shape)
print(df_merged_added_features.columns)


train, test = train_test_split(df_merged_added_features, test_size=0.2)

train_data = train.values
test_data = test.values

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# model
model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7, colsample_bytree=0.7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('Results for merged dataframe with added features')
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
print("R2: %.2f" % r2)
print("MAE: %.2f" % mae)
print('\n')



################################################################################################################################
# Christina's dataframe - merged data 

# READ THE pcl FILE
df_merged = pd.read_pickle(
    'C:/Users/nickg/Desktop/Hackathon/final_dataframe.pkl')


#Basic df elements
# print(df_merged.head(10))
# print(df_merged.shape)
# print(df_merged.columns)

#speed to index -1 
df_speed = df_merged['speed']
df_merged = df_merged.drop(['speed'], axis = 1)
df_merged['speed'] = df_speed

#drop key
df_merged = df_merged.drop(['key'], axis = 1)
print(df_merged.head(10))
print(df_merged.shape)
print(df_merged.columns)

train, test = train_test_split(df_merged, test_size=0.2)

train_data = train.values
test_data = test.values

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# model
model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7, colsample_bytree=0.7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('Results for merged dataframe')
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
print("R2: %.2f" % r2)
print("MAE: %.2f" % mae)

end = time.time()
print(f'Experiment time:{end - start} seconds')


