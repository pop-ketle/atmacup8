import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./features/train.csv')

names = ['LEGO Marvel Super Heroes', 'Ratatouille', 'Cars', 'The LEGO Movie Videogame', 'Lego Batman 3: Beyond Gotham']
for name in names:
    platforms, sales = [], []

    queried = train.query(f'Name=="{name}"')

    for i in range(len(queried)):
        row = queried.iloc[i]
#     print(queried['Platform'])
#     print(len(queried))
#     # print(queried)
#     for platform in queried['Platform'].tolist():
#         platforms.append(platform)
# print(set(platforms))