
import pandas as pd
import numpy as np


# testdata 폴더 : Consumo_cerveja.csv
# Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
#
# feature : Temperatura Media (C) : 평균 기온(C)
#
#             Precipitacao (mm) : 강수(mm)
#
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오
#
# . 을 ,로 수정
data2= pd.read_csv("testdata/Consumo_cerveja.csv").dropna()

data2 = data2.loc[:,['Temperatura Media (C)','Precipitacao (mm)','Consumo de cerveja (litros)']]
data2.columns = ['평균기온','강수량','맥주소비량']
data2['평균기온']=data2['평균기온'].str.replace(',','.')
data2['강수량']=data2['강수량'].str.replace(',','.')

print(data2.info())

# data2['평균기온']=data2['평균기온'].apply(pd.to_numeric)
data2= data2.astype({'평균기온':'float'})
# data2['강수량']=data2['강수량'].apply(pd.to_numeric)
data2= data2.astype({'강수량':'float'})
# print(data2.columns)
print(data2[:3])
# print(data2['Temperatura Media (C)'])
# print(data2['Precipitacao (mm)'])
# print(data2['Consumo de cerveja (litros)'])
# print(data2)
#

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data2, test_size = 0.3,random_state=12) # 7 : 3 random_state=12 랜덤값 고정으로 주는 함수
print(len(train_set),len(test_set)) # 255 110
# print(train_set)





from sklearn.linear_model import LinearRegression
x = data2[['평균기온','강수량']]   # feature는 2차원 배열
y = data2['맥주소비량']
# 학습은 train dataset 으로 작업
model_linear = LinearRegression().fit(X=train_set[['평균기온','강수량']], y=train_set['맥주소비량'])
print('slope : ', model_linear.coef_)  #[ 0.83966574 -0.08682071]
print('bias : ', model_linear.intercept_)  # 7.980046477148694

# 모델 평가는 test dataset 으로 작업
# print(test_set)
pred = model_linear.predict(test_set[['평균기온','강수량']])
print('예측값 : ', np.round(pred[:5].flatten(),1)) # 예측값 :  [25.5 25.4 23.  21.5 25.6]
print('실제값 : ', test_set['맥주소비량'][:5].values.flatten())
#예측값 :  실제값 :  [24.213 26.021 21.406 20.681 24.867]

from sklearn.metrics import r2_score
print('r2_score(결정계수):{}'.format(r2_score(test_set['맥주소비량'], pred)))  
# 0.3325( 결정계수) 33.25% 설명력을 가진다.

new_x = [[11.0, 0.0],[38.0, 20.0]]
new_pred = model_linear.predict(new_x)
print('새로운 값 예측 결과 : ', new_pred.flatten())  # 차원 축소함
print('새로운 값 예측 결과 : ', new_pred.ravel())    # 차원 축소함
import numpy as np
print('새로운 값 예측 결과 : ', np.squeeze(new_pred)) # 차원 축소함




