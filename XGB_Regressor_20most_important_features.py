# import libraries

import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

start = time.time()

# READ THE pcl FILES
df_mvnx_merged_data_inference_preprocessed = pd.read_pickle(
    '/home/nikolaos/Desktop/Hackathon RAIS/mvnx_merged_data_inference_preprocessed.pkl')
df_mvnx_merged_data_train_validation_test_preprocessed = pd.read_pickle(
    '/home/nikolaos/Desktop/Hackathon RAIS/mvnx_merged_data_train_validation_test_preprocessed.pkl')
#renaming
df2 = df_mvnx_merged_data_train_validation_test_preprocessed
# dropping dublicates
df_dropped_dublicates = df2.drop_duplicates()

# the sensors that correspond to our top 20 variables --> LeftLowerLeg, RightHand, Pelvis, L5, LeftUpperLeg,
#                                                  LeftShoulder, LeftFoot, Head,LeftUpperArm, RightForeArm
df_most_20imp_feat = df_dropped_dublicates.loc[:, ['id', 'LeftLowerLeg_pos_2', 'RightHand_acc_1', 'LeftLowerLeg_acc_2',
        'Pelvis_acc_2', 'LeftFoot_pos_2', 'L5_pos_2', 'LeftUpperLeg_ori_1',
        'LeftShoulder_acc_2', 'LeftLowerLeg_acc_1', 'L5_angular_acc_2', 'LeftLowerLeg_ori_2',
        'LeftShoulder_angular_acc_0', 'Head_acc_1', 'LeftUpperArm_acc_2',
        'RightHand_acc_2', 'LeftShoulder_acc_0', "Head_angular_vel_1",
        "RightForeArm_acc_1", "LeftFoot_acc_2", "RightForeArm_acc_1", "speed"]]

#shape check
print(df_most_20imp_feat.shape)

# train_test_split_per_user
def train_test_split_per_user(data, train_size=0.7):
    users = list(set(data.id))
    users = sorted(users, reverse=True)  # fix randomness
    total_users = len(users)
    slice = int(train_size * total_users)
    users_train = users[:slice]
    users_test = users[slice:]
    return data[data.id.isin(users_train)], data[data.id.isin(users_test)]


train_data, test_data = train_test_split_per_user(df_most_20imp_feat) # each id is included either in train or test set


fold_groups = train_data.id
train_data = train_data.drop(columns=['id'])
test_data = test_data.drop(columns=['id'])

train_data = train_data.values
test_data = test_data.values

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# model to gpu
model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7, colsample_bytree=0.7, tree_method='gpu_hist', gpu_id=0)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
print("R2: %.2f" % r2_score)
print("MAE: %.2f" % mae)


end = time.time()
print(end - start)


