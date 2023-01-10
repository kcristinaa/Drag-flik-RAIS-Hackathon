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

# READ THE pcl FILE
df_merged_added_features = pd.read_pickle(
    'C:/Users/nickg/Desktop/Hackathon/final_without_NaNs_df.pkl')

#Basic df elements
print(df_merged_added_features.head(2))
print(df_merged_added_features.shape)
print(df_merged_added_features.columns)

#speed to index -1 
df_speed = df_merged_added_features['speed']
df_merged_added_features = df_merged_added_features.drop(['speed'], axis = 1)
df_merged_added_features['speed'] = df_speed

#drop key
df_merged_added_features = df_merged_added_features.drop(['key'], axis = 1)
print(df_merged_added_features.head(2))
print(df_merged_added_features.shape)
print(df_merged_added_features.columns)

#column names to list
feature_names = list(df_merged_added_features.columns)
feature_names = feature_names[:841]
print(len(feature_names))


train, test = train_test_split(df_merged_added_features, test_size=0.2)
train_data = train.values
test_data = test.values
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
#################################################################
#Sort dictionary by value function 
def sort_dict_by_value(d, reverse = False):
  return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

########################################
#FI pipeline 1
# model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7, colsample_bytree=0.7).fit(X_train, y_train)

# X_train = pd.DataFrame(X_train)
# for col,score in zip(X_train.columns,model.feature_importances_):
#     print(col,score)
# X_train = X_train.values

######################################################################################
# #FI pipeline 2

# xbg_reg = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7,colsample_bytree=0.7).fit(X_train, y_train)
# xbg_reg.get_booster().get_score(importance_type='gain')
# f_importance = xbg_reg.get_booster().get_score(importance_type='gain')
# print(f_importance)
# f_importance = sort_dict_by_value(f_importance, True)
# # print(f_importance)
# importance_df = pd.DataFrame.from_dict(data=f_importance, 
#                                     orient='index')
# importance_df_top_20 = importance_df.head(20)        

# importance_df_top_20.plot.bar()
# plt.tight_layout()
# plt.show()

######################################################################################
#FI pipeline 3
model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7,colsample_bytree=0.7).fit(X_train, y_train)
model.get_booster().feature_names = feature_names
# xgboost.plot_importance(model.get_booster())
# plt.tight_layout()
# plt.show()

f_importance = model.get_booster().get_score(importance_type='gain')
# print(f_importance)
f_importance = sort_dict_by_value(f_importance, True)
# print(f_importance)
importance_df = pd.DataFrame.from_dict(data=f_importance, 
                                    orient='index')
importance_df_top_20 = importance_df.head(20)        

importance_df_top_20.plot.bar()
plt.tight_layout()
plt.show()


# y_pred = xbg_reg.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2_score = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % (mse**(1/2.0)))
# print("R2: %.2f" % r2_score)
# print("MAE: %.2f" % mae)



################################################################################################################################

# # READ THE pcl FILE
# df_merged = pd.read_pickle(
#     'C:/Users/nickg/Desktop/Hackathon/final_dataframe.pkl')

# # the sensors that correspond to our top 10 variables --> LeftUpperArm, LeftLowerLeg, LeftShoulder, RightForeArm, RightLowerLeg
#                                                         # LeftFoot, RightHand, RightFoot, T12, RightUpperArm
# df_merged_most_10imp_feat = df_merged.loc[:, ['LeftUpperArm_angular_acc_2std', 'LeftLowerLeg_pos_2mean',
#                                        'RightHand_vel_1std', "LeftFoot_vel_1std", 'LeftShoulder_acc_1std',
#                                         'RightUpperArm_pos_2mean', 'T12_pos_2mean', 'RightFoot_vel_1std',
#                                        'RightLowerLeg_pos_2mean', 'RightForeArm_vel_1std', 'speed']]

# print(df_merged_most_10imp_feat.head(2))
# # print(df_most_10imp_feat.shape)

# train, test = train_test_split(df_merged_most_10imp_feat, test_size=0.2)

# train_data = train.values
# test_data = test.values

# X_train, y_train = train_data[:, :-1], train_data[:, -1]
# X_test, y_test = test_data[:, :-1], test_data[:, -1]

# # model
# model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7, colsample_bytree=0.7)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2_score = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % (mse**(1/2.0)))
# print("R2: %.2f" % r2_score)
# print("MAE: %.2f" % mae)


# # Results
# # MSE: 28.61
# # RMSE: 5.35
# # R2: 0.72
# # MAE: 3.83

# end = time.time()
# print(end - start)


