import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
warnings.filterwarnings('ignore')

def rmsle(preds, data):
    y_true = data.get_label()
    score = np.sqrt(mean_squared_log_error(y_true, preds))
    return 'RMSLE', score, False

# def rmse(preds, data):
#     y_true = data.get_label()
#     y_true = [np.log1p(t) for t in y_true]
#     score = np.sqrt(mean_squared_error(y_true, preds))
#     return 'RMSE', score, False


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

# 型変換
train = train.astype({'Year_of_Release': str})
test  = test.astype({'Year_of_Release': str})


# 処理をまとめてやるためにtrainとtestを結合
train_length = len(train) # あとで分離するように保存
train_test   = pd.concat([train, test], ignore_index=True) # testがtrainと被ってるのでindexを再定義してる

obj_col = train.select_dtypes(include=object).columns.tolist()
print(obj_col)
for col in obj_col:
    print(col)
    le = LabelEncoder()
    encoded = le.fit_transform(train_test[col].values)
    decoded = le.inverse_transform(encoded)
    train_test[col] = encoded

# 目的変数の取得と、とりあえずtestにないデータはドロップ
y = train['Global_Sales']
train = train.drop(['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'], axis=1)

train, test = train_test[:train_length], train_test[train_length:]

lgb_params = {
    'objective': 'regression',
    # 'learning_rate': 0.00000001,
    # 'metric': 'mse',
    'max_bin': 300,
    'learning_rate': 0.05,
    'num_leaves': 40,
    # 'min_child_samples': 1,
}

scores, test_preds, valid_preds = [], [], []
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, y)):
    x_train, x_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    print(len(x_train),len(y_train))
    print(len(x_valid),len(y_valid))

    # lgb_train = lgb.Dataset(x_train, y_train)
    # lgb_valid = lgb.Dataset(x_valid, y_valid)

    # evals_result = {}
    # model = lgb.train(
    #     lgb_params, lgb_train,
    #     valid_sets=[lgb_train, lgb_valid],
    #     verbose_eval=10,
    #     num_boost_round=10000,
    #     early_stopping_rounds=10,
    #     evals_result=evals_result, # メトリックの履歴を残すオブジェクト
    #     feval=rmsle, # 独自メトリックを計算する関数
    # )
    # gbm_valid_pred = model.predict(x_valid, num_iteration=model.best_iteration)
    # gbm_test_pred  = model.predict(test, num_iteration=model.best_iteration)

    # score = np.sqrt(mean_squared_log_error(y_valid, gbm_valid_pred))
    # print(f'valid gbm acc: {score}')
    # scores.append(score)
    # test_preds.append(gbm_test_pred)
    # print(gbm_test_pred)

    train_data = Pool(x_train, y_train, cat_features=obj_col)
    valid_data = Pool(x_valid, y_valid, cat_features=obj_col)

    cab_params = {
        'eval_metric': 'MSLE',
        # 'loss_function': 'MSLE',
        'random_seed': RANDOM_SEED,
        'num_boost_round': 10000,
    }
    model = CatBoostRegressor(**cab_params)
    # model = CatBoostClassifier(eval_metric='Logloss',
    #                     num_boost_round=10000,
    #                     random_seed=RANDOM_SEED)
    model.fit(train_data, 
            eval_set=valid_data,
            early_stopping_rounds=10,
            verbose=True,
            use_best_model=True)

    cab_valid_pred = model.predict(x_valid)
    cab_test_pred  = model.predict(test)

    score = np.sqrt(mean_squared_log_error(y_valid, cab_valid_pred))
    print(f'valid cab acc: {score}')
    scores.append(score)
    print(cab_test_pred)
    test_preds.append(cab_test_pred)
    valid_preds.append(cab_valid_pred)

for score in scores: print(score)

cv = sum(scores)/len(scores)
print(f'CV: {cv}')

valid_preds = np.mean(valid_preds, axis=0)

plt.figure(figsize=(5, 5))
plt.plot([-200, 8000], [-200, 8000], color='black')
plt.scatter(y_valid, cab_valid_pred, alpha=0.2)
plt.xlim(-100, 4000)
plt.ylim(-100, 4000)
plt.xlabel('True')
plt.ylabel('Pred')
plt.show()
exit()

# submit.csvの作成
test_preds = np.mean(test_preds, axis=0)
print(test_preds)

preds = np.round(test_preds)
print(preds)

submit_df = pd.DataFrame({'Global_Sales': preds})
submit_df.to_csv(f'./submission/sample_cv:{cv}.csv', index=False)
