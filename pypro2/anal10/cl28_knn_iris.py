#  다항분류 가능 - iris dataset : 활성화 함수는 softmax 사용
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 표준화 지원 클래스

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


iris = datasets.load_iris()
# print(iris.DESCR)
# print(iris.keys())

# x = iris.data
# print(x)
# print(iris.target)


print(iris.feature_names)
# 상관관계 확인
print(np.corrcoef(iris.data[:, 2],iris.data[:,3]))
#'petal length (cm)','petal width (cm)' 만 참여
x = iris.data[:,[2,3]] # 2차원
y = iris.target # 1차원

print(x[:3])
# print(y[:3],set(y))

# train / test 분리 = 오버피팅 방지 목적
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape )

"""
# 데이터 표준화(scaling) : 데이터 크기를 , 값의 범위(scale)를 평균 0, 분산 1이 되도록 바꿔주는 것.
# 데이터 최적화 과정에서 안정성, 수렴속도를 향상. 과적합/과소적합 방지에도 효과적...

# 독립변수(feature)에 대해 작업을 한다.
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)

x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])

# 표준화 원복
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
"""

print('분류 모델 생성 --------------')
# print(x_test)



from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 3)

print(model)
model.fit(x_train,y_train) # train dataset으로 모델 학습

# 분류 예측
y_pred = model.predict(x_test) # train dataset으로 모델 검정
print('예측값 : ',y_pred)
print('실제값 : ',y_test)
print('총 갯수%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))

print('분류 정확도 확인 1')
print('accuracy:%.3f'%accuracy_score(y_test, y_pred))

print('분류 정확도 확인 2')
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print('accuracy:',(con_mat[0][0] +con_mat[1][1] +con_mat[2][2])/ len(y_test) )
# 0.9777

print('분류 정확도 확인3')
print('train : ',model.score(x_train, y_train)) # 0.9619
print('test : ',model.score(x_test, y_test)) # 0.9777

# 모델 저장

# ...

print('새로운 값으로 분류 예측')
# print(x_test[:3])
new_data = np.array([[5.1,2.4],[0.3,0.4],[3.4,0.1]]) 
# 만약 표준화하고 모델을 생셩했다면 new_data도 표준화 작업을 해야함
new_pred = model.predict(new_data)
print('예측 결과 : ',new_pred)
print('확률로 보기: ', model.predict_proba(new_data))
print('제일 큰확률로 보기: ', model.predict_proba(new_data).max()) # 확률 제일 큰값

# 시각화

plt.rc('font', family='malgun gothic')      
plt.rcParams['axes.unicode_minus']= False

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=model, test_idx=range(105, 150), title='scikit-learn제공')     




