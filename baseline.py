import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
import texthero as hero
from texthero import preprocessing
from gensim.models import word2vec
from gensim.models import KeyedVectors

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.simplefilter('ignore')

train = pd.read_csv('./features/train.csv')
test  = pd.read_csv('./features/test.csv')

# log + 1 変換
train['Global_Sales'] = np.log1p(train['Global_Sales'])

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

# 同じNameのが出てる->プラットフォームで売り上げが分散する可能性？ 'Name'の出現回数を数えて特徴量にする
train_test['Name'] = train_test['Name'].fillna('No_Title') # NameがNaNのものがあるので'No_Title'に変換

_df = pd.DataFrame(train_test['Name'].value_counts().reset_index()).rename(columns={'index': 'Name', 'Name': 'Name_Count'})
train_test = pd.merge(train_test, _df, on='Name', how='left')

print(train_test.head())
print(train_test.columns.tolist())

# プラットフォームごとの売り上げの平均、最大、最小を計算してプラットフォームの特徴を捉える NOTE: カウントとか効きそう？ 各国ごとに特徴量を作るのは効くのか？
print(len(set(train_test['Platform'])), set(train_test['Platform']))
_df = pd.DataFrame(train_test.groupby(['Platform'])['Global_Sales'].agg(['mean', 'max', 'min']).reset_index())
_df = _df.rename(columns={'mean': 'Platform_mean', 'max': 'Platform_max', 'min': 'Platform_min'})
train_test = pd.merge(train_test, _df, on='Platform', how='left')

print(train_test)



# trainとtestに分割
# train, test = train_test[:train_length], train_test[train_length:]