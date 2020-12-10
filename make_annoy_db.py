import numpy as np
from annoy import AnnoyIndex

train_embeddings = np.load('./features/platform_genre_name_train_sentence_vectors.npy')
test_embeddings  = np.load('./features/platform_genre_name_test_sentence_vectors.npy')
train_test_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)

annoy_db = AnnoyIndex(train_test_embeddings.shape[1], metric='euclidean') # annoyのdbを作る

for i, embeddings in enumerate(train_test_embeddings):
    annoy_db.add_item(i, embeddings)

annoy_db.build(n_trees=20) # annoyのビルド
annoy_db.save('./features/annoy_db.ann') # annoyのセーブ


