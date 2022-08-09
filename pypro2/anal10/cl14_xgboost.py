# Boosting이란 약한 분류기를 결합하여 강한 분류기를 만드는 과정이다. 
# 분류기 A, B, C 가 있고, 각각의 0.3 정도의 accuracy를 보여준다고 하자. 
# A, B, C를 결합하여 더 높은 정확도, 예를 들어 0.7 정도의 
# accuracy를 얻는 게 앙상블 알고리즘의 기본 원리다. 
# Boosting은 이 과정을 순차적으로 실행한다.
# A 분류기를 만든 후, 그 정보를 바탕으로 B 분류기를 만들고, 
# 다시 그 정보를 바탕으로 C 분류기를 만든다. 
# 그리고 최종적으로 만들어진 분류기들을 모두 결합하여 
# 최종 모델을 만드는 것이 Boosting의 원리다. 
# 대표적인 알고리즘으로 Adaboost(Adaptive Boosting)가 있다. 
# Adaboost는 ensemble-based classifier의 일종으로 
# weak classifier를 반복적으로 적용해서, data의 특징을 찾아가는 알고리즘이다.
# 최근에는 XGBoost가 인기 있다.

# pip install xgboost
# pip install lightgbm

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
# xgboost 보다 연산량이 적고 손실을 줄임 대용량에 적합(적으면 오버핏위험)
from lightgbm import LGBMClassifier 
import matplotlib.pyplot as plt

dataset = load_breast_cancer()
x_feature = dataset.data
y_label = dataset.target
cancer_df = pd.DataFrame(data = x_feature, columns = dataset.feature_names)

# pd.set_option('display.max_columns',None) # ... 숨겨진것도 다 나오게 출력
print(cancer_df.head(2), cancer_df.shape)
print(dataset.target_names)
print(y_label[:3]) 
print(np.sum(y_label == 0)) # 양성 212
print(np.sum(y_label == 1)) # 음성 357

x_train, x_test, y_train, y_test = train_test_split(x_feature,y_label, test_size=0.2, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = xgb.XGBClassifier(booster='gbtree', max_depth=6, n_estimators=500).fit(x_train,y_train)
# model = LGBMClassifier(booster='gbtree', max_depth=6, n_estimators=500).fit(x_train,y_train)
print(model)

pred = model.predict(x_test)
print('예측값 :', pred[:10])
print('실제값 : ', y_test[:10])

from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('정확도 : ',acc)
# 정확도 :  0.9473684210526315
print()

cl_rep = metrics.classification_report(y_test, pred)
print(cl_rep)

# 시각화 XGBClassifier 에서만 사용 가능
fig, ax = plt.subplots(figsize=(10,12)) # 칼럼을 f숫자 으로 출력
plot_importance(model,ax = ax)
plt.show()