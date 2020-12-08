import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_columns', 50)

train = pd.read_csv('./features/train.csv')
test  = pd.read_csv('./features/test.csv')

same_pub_dev = train[train['Publisher']==train['Developer']]
diff_pub_dev = train[train['Publisher']!=train['Developer']]

# 予測値の可視化
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(np.log1p(same_pub_dev['Global_Sales']), label=f'same: {len(same_pub_dev)}')
sns.distplot(np.log1p(diff_pub_dev['Global_Sales']), label=f'different: {len(diff_pub_dev)}')
ax.legend()
ax.grid()
plt.savefig(f'./figs/difference_of_the_pub_and_dev.png')
plt.show()
