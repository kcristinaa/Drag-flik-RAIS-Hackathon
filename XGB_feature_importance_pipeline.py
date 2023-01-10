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
# print(df_merged_added_features.head(2))
# print(df_merged_added_features.shape)
# print(df_merged_added_features.columns)

#speed to index -1 
df_speed = df_merged_added_features['speed']
df_merged_added_features = df_merged_added_features.drop(['speed'], axis = 1)
df_merged_added_features['speed'] = df_speed

#drop key
df_merged_added_features = df_merged_added_features.drop(['key'], axis = 1)
# print(df_merged_added_features.head(2))
# print(df_merged_added_features.shape)
# print(df_merged_added_features.columns)

#column names to list
feature_names = list(df_merged_added_features.columns)
feature_names = feature_names[:len(feature_names)-1]
# print(len(feature_names))


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

#train and test data 
train, test = train_test_split(df_merged_added_features, test_size=0.2)
train_data = train.values
test_data = test.values
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

#get the most important features in descending order 
model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7,colsample_bytree=0.7).fit(X_train, y_train)
model.get_booster().feature_names = feature_names
f_importance = model.get_booster().get_score(importance_type='gain')
f_importance = sort_dict_by_value(f_importance, True)

# plot the most important features in descending order
importance_df = pd.DataFrame.from_dict(data=f_importance, 
                                    orient='index')
importance_df_top_10 = importance_df.head(10) 
importance_df_top_10.plot.bar()
plt.tight_layout()
plt.legend('I',loc='best',prop={'size': 12})
plt.title('Feature Importance')
plt.show()

#create a list of the features in descending order
importance_df_top_10 = importance_df_top_10.T
top10 = list(importance_df_top_10.columns)
# print(top10)

#############################################################################


# Model with the top () features 
# Beacause of the stochastic nature of the XGB we iterated several times the FI pipeline 3 and we selected the following 

#those columns correspond to the sensors : L3, Right Upper Leg, Left Upper Arm, Left Shoulder, Right Foot, Right Shoulder, Right Foot,Right Toe, Right Upper Arm 
#those columns correspond to the extra added features : max knee angle 
df_merged_added_features_10_most_imp_feat = df_merged_added_features.loc[:, ['max_knee_angle', 'RightUpperLeg_pos_2mean',
                                       'L3_pos_2mean', "LeftUpperArm_angular_acc_2std", 'LeftShoulder_vel_1std',
                                        'RightShoulder_vel_1std', 'RightFoot_vel_1mean', 'LeftShoulder_acc_1std',
                                       'RightUpperArm_pos_2mean', 'RightToe_vel_1mean', 'speed']]
# print(df_merged_added_features_10_most_imp_feat.head(10))
# print(df_merged_added_features_10_most_imp_feat.shape)
# print(df_merged_added_features_10_most_imp_feat.columns)


train, test = train_test_split(df_merged_added_features_10_most_imp_feat, test_size=0.2)

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


#column names to list
feature_names = list(df_merged.columns)
feature_names = feature_names[:len(feature_names)-1]
# print(len(feature_names))


######################################################################################
#FI pipeline 3

#train and test data 
train, test = train_test_split(df_merged, test_size=0.2)

train_data = train.values
test_data = test.values

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

#get the most important features in descending order 
model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01,  eta=0.1, subsample=0.7,colsample_bytree=0.7).fit(X_train, y_train)
model.get_booster().feature_names = feature_names
f_importance = model.get_booster().get_score(importance_type='gain')
f_importance = sort_dict_by_value(f_importance, True)

# plot the most important features in descending order
importance_df = pd.DataFrame.from_dict(data=f_importance, 
                                    orient='index')
importance_df_top_10 = importance_df.head(10) 
importance_df_top_10.plot.bar()
plt.tight_layout()
plt.legend('I',loc='best',prop={'size': 12})
plt.title('Feature Importance')
plt.show()

#create a list of the features in descending order
importance_df_top_10 = importance_df_top_10.T
top10 = list(importance_df_top_10.columns)
# print(top10)

#############################################################################


#those columns correspond to the sensors : L3, Right Upper Leg, Left Upper Arm, Left Shoulder, Right Foot, 
#                                           Right Shoulder, Right Foot,Right Toe, Right Upper Arm, Left Hand 
df_merged_10_most_imp_feat = df_merged.loc[:, ['LeftHand_vel_1std', 'RightUpperLeg_pos_2mean',
                                       'L3_pos_2mean', "LeftUpperArm_angular_acc_2std", 'LeftShoulder_vel_1std',
                                        'RightShoulder_vel_1std', 'RightFoot_vel_1mean', 'LeftShoulder_acc_1std',
                                       'RightUpperArm_pos_2mean', 'RightToe_vel_1mean', 'speed']]

# print(df_merged_10_most_imp_feat.head(10))
# print(df_merged_10_most_imp_feat.shape)

train, test = train_test_split(df_merged_10_most_imp_feat, test_size=0.2)

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


