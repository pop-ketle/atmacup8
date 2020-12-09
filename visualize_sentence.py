import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.offline as offline

train = pd.read_csv('./features/train.csv')
test  = pd.read_csv('./features/test.csv')
train_sentence = np.load('./features/train_sentence_vectors.npy')
test_sentence  = np.load('./features/test_sentence_vectors.npy')

train_test = pd.concat([train, test], ignore_index=True) # indexを再定義してる
train_test_sentence = np.concatenate([train_sentence, test_sentence])

train_test = train_test.fillna('none') # Genreに'nan'があるので

# tsneで次元削減
tsne = TSNE(
    n_components=2,
    random_state=72,
    perplexity=30.0,
    method='barnes_hut',
    n_iter=1000,
    verbose=2
).fit_transform(train_test_sentence)
np.save('./features/sentence_embeddings_tsne', tsne)

traces = []
for genre in set(train_test['Genre'].tolist()):
    idx = train_test.query(f'Genre=="{genre}"').index
    csv_data      = train_test.iloc[idx]
    sentence_data = np.array([tsne[i, :] for i in idx])

    trace = go.Scatter(
        x=sentence_data[:,0],
        y=sentence_data[:,1],
        # z=sentence_data[:,2],
        text=csv_data['Name'], # それぞれの名前用
        mode='markers',
        name=genre, # 右端のリスト
        marker=dict(
            sizemode='diameter',
            colorscale='Portland',
            line=dict(color='rgb(255, 255, 255)'),
            opacity=0.9,
            size=4.2
        )
    )
    traces.append(trace)

layout = dict(height=800, width=800, title='name_sentence_embeddings')
fig    = dict(data=traces, layout=layout)
offline.plot(fig, filename='name_sentence_embeddings.html')