import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
import plotly.graph_objs as go
import plotly.offline as offline
import scipy.stats

train = pd.read_csv('./features/train.csv')
test  = pd.read_csv('./features/test.csv')
train_sentence = np.load('./features/platform_genre_name_train_sentence_vectors.npy')
test_sentence  = np.load('./features/platform_genre_name_test_sentence_vectors.npy')

train_test = pd.concat([train, test], ignore_index=True) # indexを再定義してる
train_test_sentence = np.concatenate([train_sentence, test_sentence])

train_test = train_test.fillna('none') # Genreに'nan'があるので
train_test['Global_Sales'] = train_test['Global_Sales'].replace('none', 0)

# # tsneで次元削減
# tsne = TSNE(
#     n_components=2,
#     random_state=72,
#     perplexity=30.0,
#     method='barnes_hut',
#     n_iter=1000,
#     verbose=2
# ).fit_transform(train_test_sentence)
# np.save('./features/platform_genre_name_sentence_embeddings_tsne', tsne)
# tsne = np.load('./features/sentence_embeddings_tsne.npy')
tsne = np.load('./features/platform_genre_name_sentence_embeddings_tsne.npy')

traces = []
for genre in sorted(list(set(train_test['Platform'].tolist()))):
    idx = train_test.query(f'Platform=="{genre}"').index
    csv_data      = train_test.iloc[idx]
    sentence_data = np.array([tsne[i, :] for i in idx])
    dot_size      = np.array(csv_data['Global_Sales']) * 0.02

    trace = go.Scatter(
        x=sentence_data[:,0],
        y=sentence_data[:,1],
        # z=sentence_data[:,2],
        text=csv_data['Name']+'_'+csv_data['Publisher'], # それぞれの名前用
        mode='markers',
        name=genre, # 右端のリスト
        marker=dict(
            sizemode='diameter',
            colorscale='Portland',
            line=dict(color='rgb(255, 255, 255)'),
            opacity=0.9,
            size=dot_size, #4.2,
        )
    )
    traces.append(trace)

layout = dict(height=800, width=800, title='name_sentence_embeddings')
fig    = dict(data=traces, layout=layout)
offline.plot(fig, filename='genre_name_sentence_embeddings_bubble.html')