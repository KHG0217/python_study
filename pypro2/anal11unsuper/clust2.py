# iris dataset으로 군집분석

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from anal11unsuper.clust1 import row_clusters

iris = load_iris()
ir_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(ir_df.head(2))
print(ir_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']])
# dist_vec = pdist(ir_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']],metric='euclidean')
dist_vec = pdist(ir_df.loc[:, ['sepal length (cm)', 'sepal width (cm)']],metric='euclidean')
print('dist_vec : ', dist_vec)
row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist)

row_clusters = linkage(dist_vec,method='complete')
# print('row_clusters : ',row_clusters)

df = pd.DataFrame(row_clusters, columns=['군집1','군집2','거리','멤버수'])
print(df)

row_dend = dendrogram(row_clusters)
plt.ylabel('유클리드거리')
plt.show()