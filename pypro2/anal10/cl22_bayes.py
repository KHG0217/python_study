# 나이브베이즈 분류 모델 (베이즈 통계와 생성모델에 기반한 모델)
# 특성들 사이의 독립을 가정하는 베이즈 정리를 적용한 확률 분류기

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# x = np.array([[1],[2],[3],[4],[5]])
x = np.array([1,2,3,4,5])
x = x[:, np.newaxis] 
print(x.shape)
print(x)

y = np.array([1,3,5,7,9])
print(y)

model = GaussianNB().fit(x,y)   # P(L|Feature)    P(y|x)
print(model)
pred = model.predict(x)
print('예측값 : ', pred)
print('실제값 :', y)
print('분류 정확도 : ', metrics.accuracy_score(y,pred))

# 새로운 값으로 예측 
new_x = np.array([[0.5],[2],[9],[0.1]])
new_pred = model.predict(new_x)

print('새로운 예측 결과 :',new_pred) # 새로운 예측 결과 : [1 3 9 1]
print('-'*100)

print('one-hot처리: 단어 집합 크기를 벡터차원으로 하고 0과 1로 벡터를 표현하는 방식')
print('방법1 - np.eye()')
x ='1,2,3,4,5'
x = x.split(',')
print(x)

x = np.eye(len(x))
print(x)
print()

y = np.array([1,3,5,7,9])
model = GaussianNB().fit(x,y) # x 값이 one-hot 인코딩 값
pred = model.predict(x)
print('예측값 : ', pred)
print('실제값 :', y)

print('-'*100)

print('방법2 - OneHotEncoder')
x ='1,2,3,4,5'
x = x.split(',')
print(x)

x = np.array(x)
x = x[:, np.newaxis]
one_hot = OneHotEncoder(categories = 'auto')
x= one_hot.fit_transform(x).toarray()
print(x)

y = np.array([1,3,5,7,9])
model = GaussianNB().fit(x,y) # x 값이 one-hot 인코딩 값
pred = model.predict(x)
print('예측값 : ', pred)
print('실제값 :', y)

print('new_x : 새로운 값으로 분류 예측 ---')
new_x = '0.7,2,3,4,-1.3'
new_x = new_x.split(',')
new_x = np.array(new_x)
new_x = new_x[:, np.newaxis]
print(new_x)

one_hot = OneHotEncoder(categories ='auto')
new_x = one_hot.fit_transform(new_x).toarray() # one -hot 인코딩
print(new_x)

pred2 = model.predict(new_x) #학습밧을 one - hot 으로 했으므로
print(pred2)