from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/patient.csv")
print(df.head(3))
# 종속변수 STA(환자 생사 여부)에 영향을 주는 주요 변수들을 이용해 검정 후에 해석하시오.
df_x = df[['ID','AGE','SEX','RACE','SER','CAN','CRN','INF','CPR','HRA']]
df_y = df['STA']
train_x, test_x, train_y, test_y = train_test_split(df_x,df_y, random_state= 12) # test_size = 0.25 기본값
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
#(150, 10) (50, 10) (150,) (50,)

# model
model = RandomForestClassifier(criterion ='entropy', n_estimators = 100)
model = model.fit(train_x, train_y) # 학습

# pred
pred = model.predict(test_x)
print('예측값 : ', pred[:10])
print('실제값 : ', np.array(test_y[:10]))

# 분류 정확도
print('acc : ', accuracy_score(test_y, pred)) # 0.84
acc = accuracy_score(test_y, pred)

# 교차검증 모델
cross_vali = cross_val_score(model, df_x, df_y, cv = 5)
print(cross_vali, '평균 : ',np.round(np.mean(cross_vali),3))# 평균 :  0.84


print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))
# [0.10902831 0.3045067  0.01733597 0.00702478 0.02237704 0.17072939
# 0.03368189 0.11252303 0.13051282 0.09228009]
# 영향을 주는 독립변수: AGE,CAN

def plot_feature_importances(model):   # 특성 중요도 시각화
    n_features = df_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df_x.columns)
    plt.xlabel("환자 생사 여부 영향수치")
    plt.ylabel("환자 생사 여부 영향요소")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()

 

plot_feature_importances(model) 