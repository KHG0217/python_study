# 나이브베이즈 분류기로 비 여부 분류 모델 작성
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics._scorer import accuracy_scorer

df = pd.read_csv("../testdata/weather.csv")
print(df.head(2))
print(df.columns)

x = df[['MinTemp','MaxTemp','Rainfall']]
label = df['RainTomorrow'].map({'Yes':1,'No':0})
print(x[:5])
print(label[:5])

train_x,test_x,train_y,test_y = train_test_split(x, label, random_state=1)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
# (274, 3) (92, 3) (274,) (92,)

gmodel = GaussianNB().fit(train_x,train_y) # 이진법에 유리

pred = gmodel.predict(test_x)

print('예측값 ; ', pred[:10])
print('실제값 : ', test_y[:10])

# k-fold 교차검증
from sklearn import model_selection
cross_val = model_selection.cross_val_score(gmodel,x,label, cv = 7)
print(cross_val)
print(cross_val.mean()) # 0.7979

# acc
acc = sum(test_y == pred) / len(pred)
print('분류 정확도 : ',acc) #  0.869
print('분류 정확도 : ', accuracy_score(test_y,pred)) # 0.869

# 새로운 값으로 분류 예측
# print(test_x[:3])
import numpy as np
new_weather = np.array([[0, 16, 10],[10, 36, 0],[10, 36, 40]])
print(gmodel.predict(new_weather))

