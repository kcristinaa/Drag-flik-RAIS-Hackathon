import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD


def create_train_validation_test_inference():
    merged = pd.read_pickle('mvnx_merged_data.pkl')
    merged = merged.drop_duplicates()

    inference = merged[merged['speed'].isnull()]
    print(inference.head())

    train_validation_test = merged[~merged['speed'].isnull()]
    print(train_validation_test.head())

    inference.to_pickle('mvnx_merged_data_inference.pkl')
    train_validation_test.to_pickle('mvnx_merged_data_train_validation_test.pkl')


def preprocess():
    train_validation_test = pd.read_pickle('mvnx_merged_data_train_validation_test.pkl')
    inference = pd.read_pickle('mvnx_merged_data_inference.pkl')

    print(train_validation_test.isnull().mean() * 10)
    train_validation_test_preprocessed = train_validation_test.drop(
        ['gender', 'LeftToe_acc_0', 'LeftToe_acc_1', 'LeftToe_acc_2', 'LeftToe_angular_acc_0', 'LeftToe_angular_acc_1',
         'LeftToe_angular_acc_2', 'LeftToe_vel_0', 'LeftToe_vel_1', 'LeftToe_vel_2', 'LeftToe_angular_vel_0',
         'LeftToe_angular_vel_1', 'LeftToe_angular_vel_2', 'LeftToe_ori_0', 'LeftToe_ori_1', 'LeftToe_ori_2',
         'LeftToe_ori_3', 'LeftToe_pos_0', 'LeftToe_pos_1', 'LeftToe_pos_2'], axis=1)
    print(train_validation_test_preprocessed.isnull().mean() * 100)

    print(inference.isnull().mean() * 100)
    inference_preprocessed = inference.drop(
        ['gender', 'LeftToe_acc_0', 'LeftToe_acc_1', 'LeftToe_acc_2', 'LeftToe_angular_acc_0', 'LeftToe_angular_acc_1',
         'LeftToe_angular_acc_2', 'LeftToe_vel_0', 'LeftToe_vel_1', 'LeftToe_vel_2', 'LeftToe_angular_vel_0',
         'LeftToe_angular_vel_1', 'LeftToe_angular_vel_2', 'LeftToe_ori_0', 'LeftToe_ori_1', 'LeftToe_ori_2',
         'LeftToe_ori_3', 'LeftToe_pos_0', 'LeftToe_pos_1', 'LeftToe_pos_2', 'speed'], axis=1)
    inference_preprocessed = inference_preprocessed.dropna()
    print(inference_preprocessed.isnull().mean() * 100)

    inference_preprocessed.to_pickle('mvnx_merged_data_inference_preprocessed.pkl')
    train_validation_test_preprocessed.to_pickle('mvnx_merged_data_train_validation_test_preprocessed.pkl')


def decomposed(model, name):
    train_validation_test = pd.read_pickle('mvnx_merged_data_train_validation_test_preprocessed.pkl')
    train_validation_test.reset_index(drop=True, inplace=True)
    inference = pd.read_pickle('mvnx_merged_data_inference_preprocessed.pkl')
    inference.reset_index(drop=True, inplace=True)

    transformer = model.fit(train_validation_test[train_validation_test.columns.difference(['year', 'id', 'sample', 'time', 'speed'])])

    df_1 = pd.DataFrame(transformer.transform(train_validation_test[train_validation_test.columns.difference(['year', 'id', 'sample', 'time', 'speed'])]))
    df_2 = train_validation_test.loc[:, ['year', 'id', 'sample', 'time', 'speed']]
    pd.concat([df_1, df_2], axis=1).to_pickle(f'mvnx_merged_data_train_validation_test_preprocessed_{name}.pkl')

    df_1 = pd.DataFrame(transformer.transform(inference[inference.columns.difference(['year', 'id', 'sample', 'time'])]))
    df_2 = inference.loc[:, ['year', 'id', 'sample', 'time']]
    pd.concat([df_1, df_2], axis=1).to_pickle(f'mvnx_merged_data_inference_preprocessed_{name}.pkl')


# create_train_validation_test_inference()
# preprocess()

# pca = PCA(n_components=200)
# decomposed(pca, 'pca')

truncated_svd = TruncatedSVD(n_components=200)
decomposed(truncated_svd, 'truncated_svd')
