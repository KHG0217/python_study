# 분류분석 - Decision Tree -
# - Decision Tree는 여러가지 규칙을 순차적으로 적응하면서 독립 변수 공간을 분류하는 분류 모델
# 분류와 회귀분석에 모두 사용될 수 있다. 해석이 쉽다.
# - 비모수 검정 : 선형성, 정규성, 등분산성 가정 필요없음
# - 단점 : 유의수준 판단 기준없음, 비속성/ 선형성 또는 주효과 결여/ 비안전성
# 자료에만 의존하므로 새로운 자료의 예측에서는 불안정할 수 있음

import collections
from sklearn import tree

x = [[180, 15],[177, 42],[156, 35],[174, 5],[166, 33],
     [180, 75],[167, 2],[166, 35],[174, 25],[168, 24]]
y = ['man','woman','woman','man','woman',
     'man','man','man','man','woman']
label_names = ['height','hair length']

model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3 ,random_state=0)
print(model)
fit = model.fit(x,y)
print('훈련 정확도 : ', model.score(x, y))

pred= model.predict(x)
print('예측값 : ',pred)
print('실제값 : ',y)

# 시각화
# https://www.npackd.org/p/org.graphviz.Graphviz/2.38
# 설치가 끝나면 예를 들어 C:\Graphviz\bin 폴더를 path 설정한다.
# pip install pydotplus
# pip install graphviz
import pydotplus
dot_data = tree.export_graphviz(model,feature_names = label_names,
                                out_file=None, filled=True, rounded=True )
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('red', 'orange')
edges = collections.defaultdict(list) # list type 변수 준비

for e in graph.get_edge_list():
    edges[e.get_source()].append(int(e.get_destination()))
    
for e in edges:
    edges[e].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[e][i]))[0]
        dest.set_fillcolor(colors[i])    

graph.write_png('tree.png')

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
img = imread('tree.png')
plt.imshow(img)
plt.show()

new_pred =model.predict([[170,120]])
print('새 예측값 : ', new_pred)