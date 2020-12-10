import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
# from optuna.integration.lightgbm import LightGBMTunerCV as lightgbm_tuner
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
embeddings_tsne = np.load('./features/platform_genre_name_tsentence_embeddings_tsne.npy')
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

# 各PublisherのPlatform毎のデータ件数は以下のようにして集計しています。
plat_pivot = train_test.pivot_table(index='Publisher', columns='Platform',values='Name', aggfunc='count').reset_index()

# print(plat_pivot)
# exit()

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


print(_df)

# exit()



# # 連続するN単語を頻出順に表示する＆出現数を特徴量にする # NOTE: そんなに効いてない気がする(消すこともないっちゃないんだけど...)
# def get_top_text_ngrams(corpus, n, g , s):
#     vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
#     bag_of_words = vec.transform(corpus)
#     sum_words = bag_of_words.sum(axis=0) 
#     words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if sum_words[0, idx] > s]
#     words_freq =sorted(words_freq, key = lambda x: x[1], reverse=False)
#     return words_freq[:n]

# most_common_bi = get_top_text_ngrams(train_test.Name,10000,2,5)
# most_common_bi = dict(most_common_bi)

# train_test["num_Series"] = 0
# for i in most_common_bi:
#     idx = train_test[train_test["Name"].str.contains(i)].index
#     train_test.iloc[idx, -1] = most_common_bi[i]


print(train_test.head())
print(train_test.columns.tolist())
# exit()

lgbm_params = {
    'objective': 'rmse', # 目的関数. これの意味で最小となるようなパラメータを探します. 
    'learning_rate': 0.1, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
    'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
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
drop_column = ['Name','Platform','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','Developer','Rating','Platform_and_Genre','Platform_and_Genre_and_Binning_Year']
test = test.drop(drop_column, axis=1)


# # # parameter tuning
# lgbm_params = {"objective": "regression", "metric": "rmse", "seed": RANDOM_SEED}
# _train = train.drop(drop_column, axis=1)
# train_data = lgbm.Dataset(_train, y)
# skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
# gbm = lightgbm_tuner(lgbm_params, train_data,
#         num_boost_round=10000,
#         early_stopping_rounds=50,
#         verbose_eval=50,
#         folds=skf,
#         )
# gbm.run()
# print(gbm.best_params)
# print(gbm.best_score)
# exit()


# training data の target と同じだけのゼロ配列を用意
# float にしないと悲しい事件が起こるのでそこだけ注意
oof_pred = np.zeros_like(y, dtype=np.float)
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
sns.distplot(oof_pred, label='Out Of Fold')
ax.legend()
ax.grid()
plt.savefig(f'./figs/cv:{score}_histogram.png')
plt.show()