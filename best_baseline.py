import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
import optuna
# from optuna.integration.lightgbm import LightGBMTunerCV as lightgbm_tuner
from optuna.integration import lightgbm as lightgbm_tuner
import matplotlib.pyplot as plt
import texthero as hero
from annoy import AnnoyIndex
from texthero import preprocessing
from gensim.models import word2vec
from gensim.models import KeyedVectors
from catboost import Pool, CatBoostRegressor

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.simplefilter('ignore')

N_SPLITS    = 5
RANDOM_SEED = 72

train = pd.read_csv('./features/train.csv')
test  = pd.read_csv('./features/test.csv')

# TODO: ここ最後に回して、カラム削除もやってしまえばいいのでは
def target_encoding(train, test, target_col, y_col, validation_col):
    # 学習データ全体でカテゴリにおけるyの平均を計算
    data_tmp    = pd.DataFrame({'target': train[target_col], 'y': train[y_col]})
    target_mean = data_tmp.groupby('target')['y'].mean()
    # テストデータのカテゴリを追加
    test[f'target_enc_{target_col}'] = test[target_col].map(target_mean)

    # 変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, train.shape[0])
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
    for train_idx, test_idx in skf.split(train, train[validation_col]):
        # 学習データに対して、各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[train_idx].groupby('target')['y'].mean()
        # バリデーションデータについて、変換後の値を一時配列に格納
        tmp[test_idx] = train[target_col].iloc[test_idx].map(target_mean)
    # 返還後のデータで元の変数を置換
    train[f'target_enc_{target_col}'] = tmp

# Target Encodingを行うターゲット
for target in ['Platform', 'Genre', 'Rating']:
    target_encoding(train, test, target, 'Global_Sales', 'Publisher')

'''頭がバグったのでちょっと中止　そもそもリークするのか？ いや、もちろん可能性はあるけど...
# trainにしかない特徴量 ['EU_Sales', 'Global_Sales', 'JP_Sales', 'NA_Sales', 'Other_Sales']を使う
# リークの可能性があるので、target encodingの要領でfoldを分けて集計する
# def make_sales_portfolio(train, test, target_col, y_col, validation_col):
    # 学習データ全体でジャンルごとに正規化した各Salesの情報を得る
    # data_tmp = pd.DataFrame({'Genre': train[target_col], 'y': train[y_col]})
    # _df = data_tmp.groupby('Genre')['y'].agg(['mean', 'max', 'sum'])
    # _df = _df.add_prefix(f'{y_col}_').add_suffix(f'_std_of_{target_col}') # カラム名意味わからないけどしゃあない
    # _df = ((_df - _df.min()) / (_df.max() - _df.min())).reset_index() # ジャンルごとに正規化
    # test = pd.merge(test, _df, on='Genre', how='left')
    # # 変換後の値を格納する配列を準備
    # tmp = np.repeat(np.nan, train.shape[0])
    # skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
    # for train_idx, test_idx in skf.split(train, train[validation_col]):
    #     # 学習データに対して、各カテゴリにおける目的変数の平均を計算
    #     # _df = data_tmp.iloc[train_idx].groupby('Genre')['y'].agg(['mean', 'max', 'sum'])
    #     _df = data_tmp.iloc[train_idx].groupby('Genre')['y'].sum()
    #     # バリデーションデータについて、変換後の値を一時配列に格納
    #     tmp[test_idx] = train[target_col].iloc[test_idx].map(_df)
    #     print(_df)
    #     print(tmp, sum(tmp))
    #     exit()
# for y_col in ['EU_Sales', 'Global_Sales', 'JP_Sales', 'NA_Sales', 'Other_Sales']:
#     make_sales_portfolio(train, test, 'Genre', y_col, 'Publisher')
#     exit()
'''

# 処理をまとめてやるためにtrainとtestを結合
train_length = len(train) # あとで分離するように長さを保存
train_test   = pd.concat([train, test], ignore_index=True) # indexを再定義
# train_test   = train_test.fillna('none')

# NameのEmbeddingsをt-sneかけたものを特徴量として加える
embeddings_tsne = np.load('./features/sentence_embeddings_tsne.npy')
train_test['tsne_1'] = embeddings_tsne[:,0]
train_test['tsne_2'] = embeddings_tsne[:,1]
# Platform+Genre+NameのEmbeddingsをt-sneかけたものを特徴量として加える
embeddings_tsne = np.load('./features/platform_genre_name_sentence_embeddings_tsne.npy')
train_test['tsne_3'] = embeddings_tsne[:,0]
train_test['tsne_4'] = embeddings_tsne[:,1]

def add_tbd(df):
    '''User_Scoreのtbdを特徴量としてカラムに加える
    '''
    idx_tbd = df['User_Score']=='tbd'
    df['User_Score_is_tbd'] = idx_tbd.astype(int)
    df['User_Score'] = df['User_Score'].replace('tbd', None).astype(float)
    return df

train_test = add_tbd(train_test)

# 同じNameのが出てる->プラットフォームで売り上げが分散する可能性？ 'Name'の出現回数を数えて特徴量にする、ついでに他の特徴量もいくつかcount encoding
train_test['Name'] = train_test['Name'].fillna('No_Title') # NameがNaNのものがあるので'No_Title'に変換

# PlatformとGenreを単純に文字列として結合してCountEncodingする
train_test['Platform_and_Genre'] = train_test['Platform'] + '_' + train_test['Genre']

# PlatformとGenreを単純に文字列として結合、追加でYearをビニングしたものも文字列として結合してCountEncoding
# 'Year_of_Release'を'Platform'ごとにNaNを平均値で埋める
_df = pd.DataFrame(train_test.groupby(['Platform'])['Year_of_Release'].mean().reset_index())
_df['Year_of_Release'] = [math.ceil(year) for year in _df['Year_of_Release']] # 平均値を四捨五入して入れる
train_test['Year_of_Release_fillna'] = train_test['Year_of_Release'] # ここに'Year_of_Release'を'Platform'ごとにNaNを平均値で埋めた値を入れる
for i, row in _df.iterrows():
    platform, year = row['Platform'], row['Year_of_Release']
    fillna_data = train_test[train_test['Platform']==platform]['Year_of_Release'].fillna(year)
    train_test['Year_of_Release_fillna'][fillna_data.index.tolist()] = fillna_data

# (最大値-最小値) // 5 でbinningすることで5年単位でbinnigする
diff = train_test['Year_of_Release_fillna'].max()-train_test['Year_of_Release_fillna'].min()
train_test['Binning_Year_of_Release_fillna'] = pd.cut(train_test['Year_of_Release_fillna'], int(diff//5), labels=False)

# 列結合
train_test['Platform_and_Genre_and_Binning_Year'] = train_test['Platform_and_Genre'] + '_' + train_test['Binning_Year_of_Release_fillna'].astype(str)

def count_encoding(df, target_col):
    _df = pd.DataFrame(train_test[target_col].value_counts().reset_index()).rename(columns={'index': target_col, target_col: f'CE_{target_col}'})
    return pd.merge(df, _df, on=target_col, how='left')

for target_col in ['Name','Year_of_Release','Platform','Genre','Platform_and_Genre','Platform_and_Genre_and_Binning_Year']:
    train_test = count_encoding(train_test, target_col)

def label_encoding(df, target_col):
    le = preprocessing.LabelEncoder()
    df[f'LE_{target_col}'] = le.fit_transform(df[target_col])
    return df

train_test['Genre']  = train_test['Genre'].fillna('none') # floatとstrの比較になるので置換
for target_col in ['Year_of_Release','Platform','Genre']:
    train_test = label_encoding(train_test, target_col)

def onehot_encoding(df, target_col):
    _df = pd.get_dummies(df[target_col], dummy_na=False).add_prefix(f'OH_{target_col}=')
    return pd.concat([train_test, _df], axis=1)

for target_col in ['Year_of_Release','Platform','Genre']:
    train_test = onehot_encoding(train_test, target_col)

# プラットフォームでのジャンルごとの売り上げの平均、最大、最小、合計を計算してプラットフォームでのジャンルの特徴を捉える NOTE: カウントとか効きそう？ 各国ごとに特徴量を作るのは効くのか？
for sales in ['EU_Sales','Global_Sales','JP_Sales','NA_Sales','Other_Sales','Global_Sales']:
    # Platform
    _df = pd.DataFrame(train_test.groupby(['Platform'])[sales].agg(['mean', 'max', 'min', 'sum']).reset_index())
    _df = _df.rename(columns={'mean': f'Platform_{sales}_mean', 'max': f'Platform_{sales}_max', 'min': f'Platform_{sales}_min', 'sum': f'Platform_{sales}_sum'})
    train_test = pd.merge(train_test, _df, on='Platform', how='left')

    # Genre
    _df = pd.DataFrame(train_test.groupby(['Genre'])[sales].agg(['mean', 'max', 'min', 'sum']).reset_index())
    _df = _df.rename(columns={'mean': f'Genre_{sales}_mean', 'max': f'Genre_{sales}_max', 'min': f'Genre_{sales}_min', 'sum': f'Genre_{sales}_sum'})
    train_test = pd.merge(train_test, _df, on='Genre', how='left')

# 'Rating'の出現回数を数えて、よく売れそうな対象年齢について考える
_df = pd.DataFrame(train_test.groupby(['Rating'])['Global_Sales'].agg(['mean', 'max', 'min']).reset_index())
_df = _df.rename(columns={'mean': 'Rating_mean', 'max': 'Rating_max', 'min': 'Rating_min'})
train_test = pd.merge(train_test, _df, on='Rating', how='left')

# PlatformのPublisher数、Developer数、Genre数、User_Count数をカウントしてplatformの人気度？的なものを図る
_df = pd.DataFrame(train_test.groupby(['Platform'])['Publisher'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'Platform_Publisher_count'})
train_test = pd.merge(train_test, _df, on='Platform', how='left')

_df = pd.DataFrame(train_test.groupby(['Platform'])['Developer'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'Platform_Developer_count'})
train_test = pd.merge(train_test, _df, on='Platform', how='left')

_df = pd.DataFrame(train_test.groupby(['Platform'])['Genre'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'Platform_Genre_count'})
train_test = pd.merge(train_test, _df, on='Platform', how='left')

_df = pd.DataFrame(train_test.groupby(['Platform'])['User_Count'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'Platform_User_Count_count'})
train_test = pd.merge(train_test, _df, on='Platform', how='left')

# Platformが発売された年度の作品かどうか、発売してからの経過年数を渡す、発売年がNaNはnp.nan
_df = pd.DataFrame(train_test.groupby(['Platform'])['Year_of_Release'].agg(['min', 'max']).reset_index())
release_flag, passed_years_from_release = [0]*len(train_test), [np.nan]*len(train_test)
for i, row in _df.iterrows():
    platform, release_min, release_max = row['Platform'], row['min'], row['max']

    # データにある最小の日と同じ発売年のものにフラグを立てる
    idx = train_test[(train_test['Platform']==platform) & (train_test['Year_of_Release']==release_min)].index.tolist()
    for j in idx:
        release_flag[j] = 1
    
    # 発売してからの経過年数を渡す、発売年がNaNはnp.nan
    queried_df = train_test[train_test['Platform']==platform].dropna(subset=['Year_of_Release'])
    for j, row in queried_df.iterrows():
        passed_years_from_release[j] = row['Year_of_Release'] - release_min

_df = pd.DataFrame()
_df['release_flag'] = release_flag
_df['passed_years_from_release'] = passed_years_from_release
train_test = pd.concat([train_test, _df], axis=1)

# 数年後に違うPlatformで発売されたか？
# Nameごとに発売された回数をカウントして特徴量とする
_df = pd.DataFrame(train_test.groupby(['Name'])['Name'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'sale_count'})
train_test = pd.merge(train_test, _df, on='Name', how='left')

# 別の年に発売されたかどうか、回数をカウントして特徴量とする(HD化など、人気作は再販される可能性)
_df = pd.DataFrame(train_test.groupby(['Name'])['Year_of_Release'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'resale_count'})
train_test = pd.merge(train_test, _df, on='Name', how='left')

train_test['total_sale_count'] = train_test['sale_count'] * train_test['resale_count']

# 年ごとのクチコミの多さ
_df = pd.DataFrame(train_test.groupby(['Year_of_Release'])['User_Count'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'Year_of_Release_User_Count_count'})
train_test = pd.merge(train_test, _df, on='Year_of_Release', how='left')

_df = pd.DataFrame(train_test.groupby(['Year_of_Release'])['Critic_Count'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'Year_of_Release_Critic_Count_count'})
train_test = pd.merge(train_test, _df, on='Year_of_Release', how='left')

# PublisherとDeveloperが同じか違うかのフラグ変数、同じなら1、違うなら0
same_pub_dev_idx  = train_test[train_test['Publisher']==train_test['Developer']].index.tolist()
same_pub_dev_flag = [0]*len(train_test)
for i in same_pub_dev_idx:
    same_pub_dev_flag[i] = 1
_df = pd.DataFrame(same_pub_dev_flag, columns=['same_pub_dev_flag'])
train_test = pd.concat([train_test, _df], axis=1)

# ここのPublisherの扱い方参考 https://www.guruguru.science/competitions/13/discussions/386fb2ed-f0a6-4706-85ba-7a03fedea375/
for target_col in ['Platform','Genre','Year_of_Release']:
    # 各PublisherのPlatform毎のデータ件数は以下のようにして集計しています。
    plat_pivot = train_test.pivot_table(index='Publisher', columns=target_col,values='Name', aggfunc='count').reset_index()
    plat_pivot = plat_pivot.fillna(0) # カウントだから0がいいはず

    # 行列の標準化
    plat_pivot_std = plat_pivot.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    #主成分分析の実行
    pca = PCA()
    pca.fit(plat_pivot_std)
    # データを主成分空間に写像
    feature = pca.transform(plat_pivot_std)
    feature = pd.DataFrame(feature, columns=[f'{target_col}_Publisher_PCA{x+1}' for x in range(len(plat_pivot_std.columns))])

    feature = pd.concat([plat_pivot['Publisher'], feature], axis=1)
    train_test = pd.merge(train_test, feature, on='Publisher', how='left')

    # 各DeveloperのPlatform毎のデータ件数は以下のようにして集計しています。
    plat_pivot = train_test.pivot_table(index='Developer', columns=target_col,values='Name', aggfunc='count').reset_index()
    plat_pivot = plat_pivot.fillna(0) # カウントだから0がいいはず

    # 行列の標準化
    plat_pivot_std = plat_pivot.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    #主成分分析の実行
    pca = PCA()
    pca.fit(plat_pivot_std)
    # データを主成分空間に写像
    feature = pca.transform(plat_pivot_std)
    feature = pd.DataFrame(feature, columns=[f'{target_col}_Developer_PCA{x+1}' for x in range(len(plat_pivot_std.columns))])
    feature = pd.concat([plat_pivot['Developer'], feature], axis=1)
    train_test = pd.merge(train_test, feature, on='Developer', how='left')


# User_Score x User_Countでユーザーがつけたスコアのサムを計算
train_test['User_Score_x_User_Count'] = train_test['User_Score'] * train_test['User_Count']

# User_Countに対する処理
# # NOTE: これリークしてそう やっぱりリークしてる
# _df = pd.DataFrame(train_test.groupby(['User_Count'])['Global_Sales'].agg(['mean', 'max', 'min', 'sum']).reset_index())
# _df = _df.add_prefix('User_Count_Global_Sales_').rename(columns={'User_Count_Global_Sales_User_Count': 'User_Count'})
# train_test = pd.merge(train_test, _df, on='User_Count', how='left')

# プラットフォームごとのユーザーカウントからプラットフォームの人気度を類推する
_df = pd.DataFrame(train_test.groupby(['Platform'])['User_Count'].agg(['mean', 'max', 'min', 'sum']).reset_index())
_df = _df.add_prefix('Platform_User_Count_').rename(columns={'Platform_User_Count_Platform': 'Platform'})
train_test = pd.merge(train_test, _df, on='Platform', how='left')

# trainとtestに分割
train, test = train_test[:train_length], train_test[train_length:]

y = train['Global_Sales']
y = np.log1p(y) # log + 1 変換

# 使えなさそうなドロップするカラム TODO: ここobjectの列を列挙するように変えてもいいと思ったけど、Salesがありましたね...
drop_column = ['Name','Platform','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','Developer','Rating','Platform_and_Genre','Platform_and_Genre_and_Binning_Year']
test = test.drop(drop_column, axis=1)


# # parameter tuning
# # lgbm
# x_train,x_valid,y_train,y_valid = train_test_split(train, y, random_state=RANDOM_SEED, test_size=0.3)
# # # Publisherでfoldを割るので、trainはデータを分割した後にカラムをドロップ
# x_train, x_valid = x_train.drop(drop_column, axis=1), x_valid.drop(drop_column, axis=1)

# train_data, valid_data = lgbm.Dataset(x_train, y_train), lgbm.Dataset(x_valid, y_valid)
# lgbm_params = {'objective': 'regression', 'metric': 'rmse', 'importance_type': 'gain'}
# lgbm = lightgbm_tuner.train(lgbm_params, train_data,
#                                         valid_sets=valid_data,
#                                         num_boost_round=10000,
#                                         early_stopping_rounds=50,
#                                         verbose_eval=50,
#                                         )
# print(lgbm.params)
# print(lgbm.best_iteration)
# print(lgbm.best_score)
# {'objective': 'regression', 'metric': 'rmse', 'importance_type': 'gain', 'feature_pre_filter': False, 'lambda_l1': 1.4207841579828123e-08, 'lambda_l2': 1.9057735567448798e-08, 'num_leaves': 98, 'feature_fraction': 0.5, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20, 'num_iterations': 10000, 'early_stopping_round': 50}
# 232
# defaultdict(<class 'collections.OrderedDict'>, {'valid_0': OrderedDict([('rmse', 0.8120248602423586)])})
# exit()

# # cab
# x_train,x_valid,y_train,y_valid = train_test_split(train, y, random_state=RANDOM_SEED, test_size=0.3)
# # Publisherでfoldを割るので、trainはデータを分割した後にカラムをドロップ
# x_train = x_train.drop(drop_column, axis=1)
# x_valid = x_valid.drop(drop_column, axis=1)

# train_data = Pool(x_train, y_train)
# valid_data = Pool(x_valid, y_valid)
# def objective(trial):
#     params = {
#         'eval_metric': 'RMSE',
#         'num_boost_round': 10000,
#         'random_seed': RANDOM_SEED,
#         'depth': trial.suggest_int('depth', 4, 10),                                       
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),               
#         'random_strength': trial.suggest_int('random_strength', 0, 100),                       
#         'bagging_temperature':trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
#         'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
#         'od_wait': trial.suggest_int('od_wait', 10, 50),
#     }
#     model = CatBoostRegressor(**params)
#     model.fit(train_data, 
#             eval_set=valid_data,
#             early_stopping_rounds=50,
#             verbose=False,
#             use_best_model=True)
#     return mean_squared_error(y_valid, model.predict(x_valid)) ** .5
# study = optuna.create_study()
# study.optimize(objective, n_trials = 100)
# print(f'cab_params:{study.best_params}, best_score:{study.best_value}')
# # {'depth': 8, 'learning_rate': 0.03927623848920188, 'random_strength': 17, 'bagging_temperature': 2.1960884697219814, 'od_type': 'IncToDec', 'od_wait': 41}
# # Best is trial 15 with value: 0.8141170049482175.
# exit()

# # デフォルトパラメータ
# lgbm_params = {
#     'objective': 'rmse', # 目的関数. これの意味で最小となるようなパラメータを探します. 
#     'learning_rate': 0.1, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
#     'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
#     'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
#     'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
#     'importance_type': 'gain' # 特徴重要度計算のロジック(後述)
# }
cab_params = {
    'eval_metric': 'RMSE',
    'random_seed': RANDOM_SEED,
    'learning_rate': 0.1,
    'num_boost_round': 10000,
    'depth': 5,
}

# optunaパラメータ(boruta前)
lgbm_params = {
    # 'objective': 'regression',
    'objective': 'rmse',
    'importance_type': 'gain',
    'feature_pre_filter': False,
    'lambda_l1': 1.4207841579828123e-08,
    'lambda_l2': 1.9057735567448798e-08,
    'num_leaves': 98,
    'feature_fraction': 0.5,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'num_iterations': 10000,
    'early_stopping_round': 50
}
# cab_params = {
#     'eval_metric': 'RMSE',
#     'random_seed': RANDOM_SEED,
#     'num_boost_round': 10000,
#     'depth': 8,
#     'learning_rate': 0.03927623848920188,
#     'random_strength': 17,
#     'bagging_temperature': 2.1960884697219814,
#     'od_type': 'IncToDec',
#     'od_wait': 41
# }


# training data の target と同じだけのゼロ配列を用意
# float にしないと悲しい事件が起こるのでそこだけ注意
cab_oof_pred = np.zeros_like(y, dtype=np.float)
lgbm_oof_pred = np.zeros_like(y, dtype=np.float)
scores, models = [], []
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
# num_bins = np.int(1 + np.log2(len(train)))
# bins = pd.cut(train['Global_Sales'], bins=num_bins, labels=False)
# for i, (train_idx, valid_idx) in enumerate(skf.split(train, bins.values)):
for i, (train_idx, valid_idx) in enumerate(skf.split(train, train['Publisher'])):
    x_train, x_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Publisherでfoldを割ってるので、trainはデータを分割した後にカラムをドロップ
    x_train = x_train.drop(drop_column, axis=1)
    x_valid = x_valid.drop(drop_column, axis=1)

    train_data = Pool(x_train, y_train)
    valid_data = Pool(x_valid, y_valid)

    model = CatBoostRegressor(**cab_params)
    model.fit(train_data, 
            eval_set=valid_data,
            early_stopping_rounds=50,
            verbose=False,
            use_best_model=True)
    cab_valid_pred = model.predict(x_valid)
    score = mean_squared_error(y_valid, cab_valid_pred) ** .5
    print(f'Fold {i} CAB RMSLE: {score}')

    cab_oof_pred[valid_idx] = cab_valid_pred
    models.append(model)
    scores.append(score)

    model = lgbm.LGBMRegressor(**lgbm_params)
    model.fit(x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=50,
    )

    lgbm_valid_pred = model.predict(x_valid)
    score = mean_squared_error(y_valid, lgbm_valid_pred) ** .5
    print(f'Fold {i} LGBM RMSLE: {score}')

    lgbm_oof_pred[valid_idx] = lgbm_valid_pred
    models.append(model)
    scores.append(score)

# fold全体のスコアと、平均のスコアを出す
for i, s in enumerate(scores):
    if i%2==0:
        print(f'Fold {i} CAB RMSLE: {s}')
    else:
        print(f'Fold {i} LGBM RMSLE: {s}')

score = sum(scores) / len(scores)
print(score)

# ファイルを生成する前にワンクッション置きたい
# exit()

pred = np.array([model.predict(test) for model in models])
pred = np.mean(pred, axis=0)
pred = np.expm1(pred)
pred = np.where(pred < 0, 0, pred)
sub_df = pd.DataFrame({ 'Global_Sales': pred })
sub_df.to_csv(f'./submission/cv:{score}_sub.csv', index=False)

################################

# feature importanceの可視化
feature_importance_df = pd.DataFrame()
for i, model in enumerate(models):
    _df = pd.DataFrame()
    _df['feature_importance'] = model.feature_importances_
    _df['column'] = train.drop(drop_column, axis=1).columns
    _df['fold'] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

order = feature_importance_df.groupby('column')\
    .sum()[['feature_importance']]\
    .sort_values('feature_importance', ascending=False).index[:50]

fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
ax.tick_params(axis='x', rotation=90)
ax.grid()
fig.tight_layout()
plt.savefig(f'./figs/cv:{score}_feature_importance.png')
plt.show()

# 予測値の可視化
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(np.log1p(pred), label='Test Predict')
sns.distplot(cab_oof_pred, label='CAB Out Of Fold')
sns.distplot(lgbm_oof_pred, label='LGBM Out Of Fold')
ax.legend()
ax.grid()
plt.savefig(f'./figs/cv:{score}_histogram.png')
plt.show()