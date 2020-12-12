import os
import math
import random
import optuna
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgbm
import matplotlib.pyplot as plt
import texthero as hero
from annoy import AnnoyIndex
from texthero import preprocessing
from gensim.models import word2vec
from gensim.models import KeyedVectors
from catboost import Pool, CatBoostRegressor
from optuna.integration import lightgbm as lightgbm_tuner

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

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
train_test["Publisher"] = train_test["Publisher"].replace("Unknown", '')

# # PCAが結構効いたので名前のEmbeddingsもPCAに突っ込んでみる
# train_embeddings = np.load('./features/platform_genre_name_train_sentence_vectors.npy')
# test_embeddings  = np.load('./features/platform_genre_name_test_sentence_vectors.npy')
# train_test_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)

# # # 行列の標準化
# mm = preprocessing.MinMaxScaler()
# train_test_embeddings_std = mm.fit_transform(train_test_embeddings)

# #主成分分析の実行
# pca = PCA()
# pca.fit(train_test_embeddings_std)
# # データを主成分空間に写像
# feature = pca.transform(train_test_embeddings_std)

# # 第5主成分まで取得して、特徴量に足す
# feature = pd.DataFrame(feature, columns=[f'Name_Embeddings_PCA{x+1}' for x in range(train_test_embeddings_std.shape[1])])
# feature = feature[['Name_Embeddings_PCA1','Name_Embeddings_PCA2','Name_Embeddings_PCA3','Name_Embeddings_PCA4','Name_Embeddings_PCA5']]
# train_test = pd.concat([train_test, feature], axis=1)


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

def label_encoding(df, target_col):
    le = preprocessing.LabelEncoder()
    df[f'LE_{target_col}'] = le.fit_transform(df[target_col])
    return df

# floatとstrの比較になるので置換
train_test['Genre']  = train_test['Genre'].fillna('none')
train_test['Rating'] = train_test['Rating'].fillna('none')
train_test['Platform_and_Genre'] = train_test['Platform_and_Genre'].fillna('none')
train_test['Platform_and_Genre_and_Binning_Year'] = train_test['Platform_and_Genre_and_Binning_Year'].fillna('none')
for target_col in ['Name','Year_of_Release','Platform','Genre','Rating','Platform_and_Genre','Platform_and_Genre_and_Binning_Year']:
    train_test = count_encoding(train_test, target_col)
    train_test = label_encoding(train_test, target_col)

def onehot_encoding(df, target_col):
    _df = pd.get_dummies(df[target_col], dummy_na=False).add_prefix(f'OH_{target_col}=')
    return pd.concat([train_test, _df], axis=1)

for target_col in ['Year_of_Release','Platform','Genre']:
    train_test = onehot_encoding(train_test, target_col)

# プラットフォームでのジャンルごとの売り上げの平均、最大、最小、合計を計算してプラットフォームでのジャンルの特徴を捉える NOTE: カウントとか効きそう？ 各国ごとに特徴量を作るのは効くのか？
# Critic_Score,Critic_Count,User_Score,User_Countも加えて、評価の特徴量も加える
for sales in ['EU_Sales','Global_Sales','JP_Sales','NA_Sales','Other_Sales','Global_Sales','Critic_Score','Critic_Count','User_Score','User_Count']:
    # Platform
    _df = pd.DataFrame(train_test.groupby(['Platform'])[sales].agg(['mean', 'max', 'min', 'sum']).reset_index())
    _df = _df.rename(columns={'mean': f'Platform_{sales}_mean', 'max': f'Platform_{sales}_max', 'min': f'Platform_{sales}_min', 'sum': f'Platform_{sales}_sum'})
    train_test = pd.merge(train_test, _df, on='Platform', how='left')

    # Genre
    _df = pd.DataFrame(train_test.groupby(['Genre'])[sales].agg(['mean', 'max', 'min', 'sum']).reset_index())
    _df = _df.rename(columns={'mean': f'Genre_{sales}_mean', 'max': f'Genre_{sales}_max', 'min': f'Genre_{sales}_min', 'sum': f'Genre_{sales}_sum'})
    train_test = pd.merge(train_test, _df, on='Genre', how='left')

    # Rating
    _df = pd.DataFrame(train_test.groupby(['Rating'])[sales].agg(['mean', 'max', 'min', 'sum']).reset_index())
    _df = _df.rename(columns={'mean': f'Rating_{sales}_mean', 'max': f'Rating_{sales}_max', 'min': f'Rating_{sales}_min', 'sum': f'Rating_{sales}_sum'})
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

_df = pd.DataFrame(train_test.groupby(['Platform'])['Critic_Count'].agg(['count']).reset_index())
_df = _df.rename(columns={'count': 'Platform_User_Critic_count'})
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

# # 同じ年に同ジャンル発売が多ければ、買うのが割れるかもしれない NOTE: 効果なかった
# _df = pd.DataFrame(train_test.groupby(['Year_of_Release','Genre'])['Genre'].agg(['count']).reset_index())
# # _df['Year_of_Release_Genre'] = _df['Year_of_Release'].astype(int).astype(str) + '_' + _df['Genre']
# _df = _df.rename(columns={'count': 'Year_Genre_count'})
# train_test = pd.merge(train_test, _df, on=['Year_of_Release','Genre'], how='left')

# ここのPublisherの扱い方参考 https://www.guruguru.science/competitions/13/discussions/386fb2ed-f0a6-4706-85ba-7a03fedea375/
for target_col in ['Platform','Genre','Year_of_Release','Rating']:
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

# Critic_Score x Critic_Countで評論家がつけたスコアのサムを計算
train_test['Critic_Score_x_Critic_Count'] = train_test['Critic_Score'] * train_test['Critic_Count']

# User_Countに対する処理
# # NOTE: これリークしてそう やっぱりリークしてる
# _df = pd.DataFrame(train_test.groupby(['User_Count'])['Global_Sales'].agg(['mean', 'max', 'min', 'sum']).reset_index())
# _df = _df.add_prefix('User_Count_Global_Sales_').rename(columns={'User_Count_Global_Sales_User_Count': 'User_Count'})
# train_test = pd.merge(train_test, _df, on='User_Count', how='left')

# # プラットフォームごとのユーザーカウントからプラットフォームの人気度を類推する # NOTE: この処理、上に入ったので
# _df = pd.DataFrame(train_test.groupby(['Platform'])['User_Count'].agg(['mean', 'max', 'min', 'sum']).reset_index())
# _df = _df.add_prefix('Platform_User_Count_').rename(columns={'Platform_User_Count_Platform': 'Platform'})
# train_test = pd.merge(train_test, _df, on='Platform', how='left')

# シリーズの特徴量作成
# def get_top_text_ngrams(corpus, n):
#     try:
#         vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
#     except ValueError:
#         return [('', '')]
#     bag_of_words = vec.transform(corpus)
#     sum_words = bag_of_words.sum(axis=0) 
#     words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#     words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
#     return words_freq[:n]

# # stopwordsの削除
# def remove_stopwords(text):
#     final_text = []
#     for i in text.split():
#         if i.strip().lower() not in stop_words:
#             if i.strip().isalpha():
#                 final_text.append(i.strip())
#     return " ".join(final_text)

# # '''近傍を使う処理、あんまりいいのが思い浮かばない
# # annoyで近傍を持ってきてシリーズを類推する
# annoy_db = AnnoyIndex(768, metric='euclidean') # shape、ハードエンコーディングだけどしらね
# annoy_db.load('./features/annoy_db.ann')

# # ベクトルvを与えると、近傍n個のアイテムを取り出せる
# # include_distancesはTrueで２地点間の距離を含める
# train_embeddings = np.load('./features/platform_genre_name_train_sentence_vectors.npy')
# test_embeddings  = np.load('./features/platform_genre_name_test_sentence_vectors.npy')
# train_test_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)

# series_titles = []
# for i, embeddings in tqdm(enumerate(train_test_embeddings), total=len(train_test_embeddings)):
#     # 1(対象)+5個近傍を集めてきて、bigramで出現頻度が最も高いものを対象のシリーズ名にする
#     nn_idxs, distances = annoy_db.get_nns_by_vector(embeddings, 6, search_k=-1, include_distances=True)

#     names = pd.Series(train_test.iloc[nn_idxs]['Name'].values.tolist()).apply(remove_stopwords)
#     most_common  = get_top_text_ngrams(names, 20)
#     series_title = most_common[0][0]
#     series_titles.append(series_title)

# # シリーズ名の出現回数が5以下のものは、シリーズと認めず'none'に置き換える(これだとシリーズ数が555になる)
# c = collections.Counter(series_titles) # 出現回数のカウンター
# replace_dict = dict()
# for series, cnt in c.most_common():
#     replace_dict[series] = 'none' if cnt<=5 else series

# series_titles = pd.DataFrame(series_titles, columns=['Series_Title'])
# series_titles = series_titles.replace((replace_dict))
# train_test = pd.concat([train_test, series_titles], axis=1)

# # 各種エンコーディングを行う
# train_test = count_encoding(train_test, 'Series_Title')
# train_test = label_encoding(train_test, 'Series_Title')
# train_test = onehot_encoding(train_test, 'Series_Title')


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

# exit()

# デフォルトパラメータ
lgbm_params = {
    'objective': 'rmse', # 目的関数. これの意味で最小となるようなパラメータを探します. 
    'learning_rate': 0.1, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
    'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
    'importance_type': 'gain' # 特徴重要度計算のロジック(後述)
}
cab_params = {
    'eval_metric': 'RMSE',
    'random_seed': RANDOM_SEED,
    'learning_rate': 0.1,
    'num_boost_round': 10000,
    'depth': 5,
}

# # optunaパラメータ
# lgbm_params = {
#     # 'objective': 'regression',
#     'metric': 'rmse',
#     'importance_type': 'gain',
#     'feature_pre_filter': False,
#     'lambda_l1': 4.444175241213668,
#     'lambda_l2': 2.2184552837713922,
#     'num_leaves': 31,
#     'feature_fraction': 0.6799999999999999,
#     'bagging_fraction': 0.9887595028155919,
#     'bagging_freq': 7,
#     'min_child_samples': 20,
#     'num_iterations': 10000,
#     'early_stopping_round': 50
# }
# cab_params = {
#     'eval_metric': 'RMSE',
#     'random_seed': RANDOM_SEED,
#     'num_boost_round': 10000,
#     'depth': 7,
#     'learning_rate': 0.019257671945706635,
#     'random_strength': 75,
#     'bagging_temperature': 25.581312249964302,
#     'od_type': 'Iter',
#     'od_wait': 46,
# }

# trainとtestに分割
train, test = train_test[:train_length], train_test[train_length:]

y = train['Global_Sales']
y = np.log1p(y) # log + 1 変換

# 文字列とかで使えなさそうなドロップするカラム TODO: ここobjectの列を列挙するように変えてもいいと思ったけど、Salesがありましたね...
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
# # {'objective': 'regression', 'metric': 'rmse', 'importance_type': 'gain', 'feature_pre_filter': False, 'lambda_l1': 4.444175241213668, 'lambda_l2': 2.2184552837713922, 'num_leaves': 31, 'feature_fraction': 0.6799999999999999, 'bagging_fraction': 0.9887595028155919, 'bagging_freq': 7, 'min_child_samples': 20, 'num_iterations': 10000, 'early_stopping_round': 50}
# # 535
# # defaultdict(<class 'collections.OrderedDict'>, {'valid_0': OrderedDict([('rmse', 0.8047787726626557)])})
# exit()

# cab
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
# # cab_params:{'depth': 7, 'learning_rate': 0.019257671945706635, 'random_strength': 75, 'bagging_temperature': 25.581312249964302, 'od_type': 'Iter', 'od_wait': 46}
# # best_score:0.818030112093115
# exit()

# training data の target と同じだけのゼロ配列を用意
# float にしないと悲しい事件が起こるのでそこだけ注意
cab_oof_pred  = np.zeros_like(y, dtype=np.float)
lgbm_oof_pred = np.zeros_like(y, dtype=np.float)
scores, models = [], []
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
# num_bins = np.int(1 + np.log2(len(train)))
# bins = pd.cut(train['Global_Sales'], bins=num_bins, labels=False)
# for i, (train_idx, valid_idx) in enumerate(skf.split(train, bins.values)):
for i, (train_idx, valid_idx) in enumerate(skf.split(train, train['Platform'])):
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
        print(f'Fold {i-1} LGBM RMSLE: {s}')

score = sum(scores) / len(scores)
print(score)

# ファイルを生成する前にワンクッション置きたい
# exit()

pred = np.array([model.predict(test) for model in models])
pred = np.mean(pred, axis=0)
pred = np.expm1(pred)
pred = np.where(pred < 0, 0, pred)
sub_df = pd.DataFrame({ 'Global_Sales': pred})
sub_df.to_csv(f'./submission/cv:{score}_sub.csv', index=False)

################################

# feature importanceの可視化
feature_importance_df = pd.DataFrame()
for i, model in enumerate(models):
    if i%2==0: continue
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