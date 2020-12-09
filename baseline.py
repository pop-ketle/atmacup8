import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
import matplotlib.pyplot as plt
import texthero as hero
from texthero import preprocessing
from gensim.models import word2vec
from gensim.models import KeyedVectors

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold

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
# print(train_test['Year_of_Release'].max(), train_test['Year_of_Release'].min()) -> 2020.0 1980.0 なので5年単位でbinnigしてみる

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

for target_col in ['Name','Year_of_Release','Platform','Platform_and_Genre','Platform_and_Genre_and_Binning_Year']:
    train_test = count_encoding(train_test, target_col)

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

# PublisherとDeveloperが同じか違うかのフラグ変数、同じなら1、違うなら0
same_pub_dev_idx  = train_test[train_test['Publisher']==train_test['Developer']].index.tolist()
same_pub_dev_flag = [0]*len(train_test)
for i in same_pub_dev_idx:
    same_pub_dev_flag[i] = 1
_df = pd.DataFrame(same_pub_dev_flag, columns=['same_pub_dev_flag'])
train_test = pd.concat([train_test, _df], axis=1)

# User_Score x User_Countでユーザーがつけたスコアのサムを計算
train_test['User_Score_x_User_Count'] = train_test['User_Score'] * train_test['User_Count']

print(train_test.head())
print(train_test.columns.tolist())
# exit()

lgbm_params = {
    'objective': 'rmse', # 目的関数. これの意味で最小となるようなパラメータを探します. 
    'learning_rate': .1, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
    'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    'colsample_bytree': .5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
    'importance_type': 'gain' # 特徴重要度計算のロジック(後述)
}

# trainとtestに分割
train, test = train_test[:train_length], train_test[train_length:]

y = train['Global_Sales']
y = np.log1p(y) # log + 1 変換

print(y)
print(train)
print(test)

# 使えなさそうなドロップするカラム TODO: ここobjectの列を列挙するように変えてもいいと思ったけど、Salesがありましたね...
drop_colunm = ['Name','Platform','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','Developer','Rating','Platform_and_Genre','Platform_and_Genre_and_Binning_Year']
test  = test.drop(drop_colunm, axis=1)

# training data の target と同じだけのゼロ配列を用意
# float にしないと悲しい事件が起こるのでそこだけ注意
oof_pred = np.zeros_like(y, dtype=np.float)
scores, models = [], []
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
for i, (train_idx, valid_idx) in enumerate(skf.split(train, train['Publisher'])):
    x_train, x_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Publisherでfoldを割ってるので、trainはデータを分割した後にカラムをドロップ
    x_train = x_train.drop(drop_colunm, axis=1)
    x_valid = x_valid.drop(drop_colunm, axis=1)

    model = lgbm.LGBMRegressor(**lgbm_params)
    model.fit(x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=50,
    )

    lgbm_valid_pred = model.predict(x_valid)
    score = mean_squared_error(y_valid, lgbm_valid_pred) ** .5
    print(f'Fold {i} RMSLE: {score}')

    oof_pred[valid_idx] = lgbm_valid_pred
    models.append(model)
    scores.append(score)

# fold全体のスコアと、平均のスコアを出す
for i, s in enumerate(scores): print(f'Fold {i} RMSLE: {s}')
score = sum(scores) / len(scores)
print(score)

# ファイルを生成する前にワンクッション置きたい
exit()

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
    _df['column'] = train.drop(drop_colunm, axis=1).columns
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
sns.distplot(oof_pred, label='Out Of Fold')
ax.legend()
ax.grid()
plt.savefig(f'./figs/cv:{score}_histogram.png')
plt.show()