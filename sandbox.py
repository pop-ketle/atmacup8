import re
import numpy as np
import pandas as pd
import lightgbm as lgb
import pandas_profiling as pdp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
warnings.filterwarnings('ignore')

def rmsle(preds, data):
    y_true = data.get_label()
    score = np.sqrt(mean_squared_log_error(y_true, preds))
    return 'RMSLE', score, False

def target_encoding(train_df, test_df, input_column_name, output_column_name):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    c = input_column_name
    # 学習データ全体で各カテゴリにおけるyの平均を計算
    data_tmp    = pd.DataFrame({c: train_df[c], 'target':train_df['Global_Sales']})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # テストデータのカテゴリを置換
    test_df[output_column_name] = test_df[c].map(target_mean)

    # 返還後の値を格納する配列を準備
    tmp = np.repeat(np.nan, train_df.shape[0])

    for i, (train_idx, test_idx) in enumerate(kf.split(train_df, train_df['Global_Sales'])):
        # 学習データについて、各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[train_idx].groupby(c)['target'].mean()
        # バリデーションデータについて、返還後の値を一時配列に格納
        tmp[test_idx] = train_df[c].iloc[test_idx].map(target_mean)
    # 変換後のデータで元の変数を置換
    train_df[output_column_name] = tmp

N_SPLITS    = 5
RANDOM_SEED = 510

train = pd.read_csv('./features/train.csv')
test  = pd.read_csv('./features/test.csv')

train = train.fillna('none')
test  = test.fillna('none')
# 欠損値'none'を'-1'に変換する(intとstrで比較できないので)
for c in ['Critic_Score', 'Critic_Count', 'User_Count']: # 'User_Score'は'tbd'があるのでなし
    train[c] = train[c].replace('none', -1)
    test[c] = test[c].replace('none', -1)

def onehot_encoding(df, target):
    mlb = MultiLabelBinarizer()
    mlb.fit([set(df[target].unique())])
    MultiLabelBinarizer(classes=None, sparse_output=False)

    new_df = mlb.transform(df.values)
    new_df = pd.DataFrame(new_df, columns=[f'{target}_{c}' for c in mlb.classes_])
    return pd.concat([df, new_df], axis=1)

obj_col = train.select_dtypes(include=object).columns.tolist()
print(obj_col)
# オブジェクトの列は全てターゲットエンコーディング実施
for col in obj_col:
    print(col)
    if col!='Name' and col!='Year_of_Release': # Year_of_Releaseは欠損値が多いので
        train = onehot_encoding(train, col)
        test  = onehot_encoding(test, col)
    target_encoding(train, test, col, "enc_"+col)
    train = train.drop(col, axis=1)
    test  = test.drop(col, axis=1)

y = train['Global_Sales']
train = train.drop(['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'], axis=1) # とりあえずtestにないデータはドロップ

# train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# test  = test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in train.columns]
test.columns  = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in test.columns]

print(train.columns.tolist())

lgb_params = {
    'objective': 'regression',
    # 'metric': 'mse',
    # 'learning_rate': 0.65,
    # 'max_bin': 300,
    # 'learning_rate': 0.05,
    # 'num_leaves': 40
}

kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, y)):
    x_train, x_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # lr = LinearRegression()
    # lr.fit(x_train.values, y_train.values)
    # y_pred = lr.predict(x_valid)
    # print(y_pred)
    # print(sorted(y_pred)[:5])
    # rmsle = mean_squared_log_error(y_valid, gbm_valid_pred) ** .5

    

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid)

    evals_result = {}
    model = lgb.train(
        lgb_params, lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        verbose_eval=10,
        num_boost_round=10000,
        early_stopping_rounds=10,
        evals_result=evals_result, # メトリックの履歴を残すオブジェクト
        feval=rmsle, # 独自メトリックを計算する関数
    )
    gbm_valid_pred = model.predict(x_valid, num_iteration=model.best_iteration)
    # rmsle = mean_squared_log_error(y_valid, gbm_valid_pred) ** .5
    # for t,y in zip(y_valid, gbm_valid_pred):
        # print(t,y)
    print(gbm_valid_pred)
    print(sorted(gbm_valid_pred)[:5])
    print(sorted(y_valid)[:5])
    # print(rmsle)
    exit()
