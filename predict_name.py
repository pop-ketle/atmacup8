# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold


# train = pd.read_csv('./features/train.csv')
# test  = pd.read_csv('./features/test.csv')
# train_name_embeddings = np.load('./features/train_sentence_vectors.npy')
# test_name_embeddings  = np.load('./features/test_sentence_vectors.npy')

# skf = StratifiedKFold(n_splits=5, random_state=72, shuffle=True)
# for i, (train_idx, valid_idx) in enumerate(skf.split(train_name_embeddings, train['Publisher'])):
#     x_train, x_valid = train_name_embeddings.iloc[train_idx], train_name_embeddings.iloc[valid_idx]
#     y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

import numpy as np
import pandas as pd
from textdistance import jaro

# train/testの読み込みと結合
train = pd.read_csv('./features/train.csv')
test = pd.read_csv('./features/test.csv')
df = pd.concat([train,test]).reset_index(drop=True)

# Nameに空欄があるとややこしくなりそうなので埋めておく
# ローデータで近接データ見てそれっぽいやつで埋めています（おそらく遠からず近からず）
df["Name"].fillna("Mortal Kombat 2",inplace=True)

# Unknownをnanに書き換え
df["Publisher"] = df["Publisher"].replace("Unknown",np.nan)

# Publisherがnanの255ゲームについて、全ゲームとの類似度を計算して格納する列を作る
names = df[df["Publisher"].isnull()].Name
for name in names:
    df["similarity_" + name] = df["Name"].map(lambda x: jaro(name, x))

# Publisherがnullのやつにフラグを付けておく
idx_null = df["Publisher"].isnull()
df["Publisher_is_null"] = idx_null.astype(int)

# Publisherがnullのインデックスを取得
idx_null = df[df["Publisher"].isnull()].index

# Publisherがnullのインデックスをループ
for i in idx_null:
    # 名前取得
    name = df.loc[i,"Name"]
    # break前提なので100に意味はないがとにかく大きい数字でループ
    for j in range(100):
        # 対象の類似度降順で並び替えて、上から順にPublisherの値を取得
        temp = df.sort_values(by="similarity_" + name,ascending=False).iloc[j,4]
        # tempがnullじゃなくなったらbreak
        if not pd.isnull(temp):
            break
    # tempで穴埋め
    df.loc[i,"Publisher"] = temp