# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

start = time.time()

# READ THE pcl FILES
df_merged = pd.read_pickle(
    'C:/Users/user1/Desktop/Drag-flik-RAIS-Hackathon-main/final_dataframe.pkl')


# the sensors that correspond to our top 10 variables --> LeftUpperArm, LeftLowerLeg, LeftShoulder, RightForeArm, RightLowerLeg
                                                        # LeftFoot, RightHand, RightFoot, T12, RightUpperArm
df_most_10imp_feat = df_merged.loc[:, ['LeftUpperArm_angular_acc_2std', 'LeftLowerLeg_pos_2mean',
                                       'RightHand_vel_1std', "LeftFoot_vel_1std", 'LeftShoulder_acc_1std',
                                        'RightUpperArm_pos_2mean', 'T12_pos_2mean', 'RightFoot_vel_1std',
                                       'RightLowerLeg_pos_2mean', 'RightForeArm_vel_1std', 'speed']]

print(df_most_10imp_feat.head(2))
# print(df_most_10imp_feat.shape)

train, test = train_test_split(df_most_10imp_feat, test_size=0.2)

train_data = train.values
test_data = test.values

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# model
model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7, colsample_bytree=0.7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
print("R2: %.2f" % r2_score)
print("MAE: %.2f" % mae)


# Results
# MSE: 28.61
# RMSE: 5.35
# R2: 0.72
# MAE: 3.83

end = time.time()
print(end - start)


