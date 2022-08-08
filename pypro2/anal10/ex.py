

from io import StringIO

from daal4py._daal4py import np
from daal4py.sklearn.linear_model.logistic_path import LogisticRegression
from sklearn.metrics._classification import accuracy_score
from sklearn.model_selection._split import train_test_split

import pandas as pd


# data =StringIO("""
# 요일,외식유무,소득수준
#
# 토,0,57
#
# 토,0,39
#
# 토,0,28
#
# 화,1,60
#
# 토,0,31
#
# 월,1,42
#
# 토,1,54
#
# 토,1,65
#
# 토,0,45
#
# 토,0,37
#
# 토,1,98
#
# 토,1,60
#
# 토,0,41
#
# 토,1,52
#
# 일,1,75
#
# 월,1,45
#
# 화,0,46
#
# 수,0,39
#
# 목,1,70
#
# 금,1,44
#
# 토,1,74
#
# 토,1,65
#
# 토,0,46
#
# 토,0,39
#
# 일,1,60
#
# 토,1,44
#
# 일,0,30
#
# 토,0,34
# """)
#
#
# # [로지스틱 분류분석 문제1]
# #
# # 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 
# #주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# #
# # 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 
# #로지스틱 회귀분석을 실시하라.
# #
# # 키보드로 소득 수준(양의 정수)을 입력하면 
# #외식 여부 분류 결과 출력하라.
#
#
# df = pd.read_csv(data)
# print(df.head(3))
# # 요일  외식유무  소득수준
#
# df =df.loc[(df['요일'] == '토') | (df['요일'] == '일')]
#
# # print(df.head)
#
# import statsmodels.formula.api as smf
# import numpy as np
# import statsmodels.api as sm
#
# # 요일  외식유무  소득수준
# formula = '외식유무 ~ 소득수준'
# model = smf.logit(formula = formula, data= df, family=sm.families.Binomial()).fit()
# # print(model)
# # print(model.summary())
# pred = model.predict()
# print('예측값 :',pred)
# print('예측값 :',np.around(pred))
# print('실제값 :',df['외식유무'].values)
#
# conf_tab =model.pred_table()
# print('confusion matrix : \n ', conf_tab)
# # confusion matrix : 
# #   [[10.  1.]   19개 예측성공 2개 실패
# #  [ 1.  9.]]
#
# a =  int(input('소득 수준(양의 정수)을 입력하세요: '))
# data = pd.Series(a)
# frame= pd.DataFrame()
# frame['소득수준']=data
#
# print(frame)
# pred2 = model.predict(frame['소득수준'])
# print('new_pred: ',np.around(pred2))
# [로지스틱 분류분석 문제2] 
#
# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
#
# 안경 : 값0(착용X), 값1(착용O)
#
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
#
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X
#
# 게임과 tv시청 으로 안경착용 유무분류
# data2= pd.read_csv("../testdata/bodycheck.csv")
# print(data2.head(3))
# #    번호  게임   신장  체중  TV시청  안경유무
# # 0   1   2  146  34     2     0
# # 1   2   6  169  57     3     1
# # 2   3   9  160  48     3     1
#
# x = data2[['게임','TV시청']].values
# print(x)
# y = data2['안경유무'].values # 1차원
# print(y)
#
# # train / test 분리 = 오버피팅 방지 목적
# x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.3, random_state=0)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape )
#
# print('분류 모델 생성 --------------')
#
# model = LogisticRegression(C = 100.0, solver ='lbfgs', multi_class='auto')
# # C 속성 : L2 규제 - 모델에 패널티 적용. 숫자 값을 조정해 가며 분류 정확도를 확인
# # 숫자가 작을수록 규제 ↓
# print(model)
# model.fit(x_train,y_train) # train dataset으로 모델 학습
# print('분류 정확도 확인 1')
# y_pred = model.predict(x_test)
# print('accuracy:%.3f'%accuracy_score(y_test, y_pred)) #1
#
# print('분류 정확도 확인3')
# print('train : ',model.score(x_train, y_train)) # 1.0
# print('test : ',model.score(x_test, y_test)) # 1.0  
#
#
# game=int(input('게임플레이 시간 입력: '))
# tv=int(input('TV시청 시간 입력: '))
#
# gdata = pd.Series(game)
# tdata = pd.Series(tv)
# frame2= pd.DataFrame()
#
# frame2['게임']=gdata
# frame2['TV시청']=tdata
#
# # print(frame2.values)
#
#
# y_pred = model.predict(frame2.values)
# print('안경착용 유무 미착용[0],착용[1] : ',y_pred)
# # print('안경착용유무 :', np.rint(model.predict(y_pred)))
# # print('안경 미착용' if np.rint(model.predict(y_pred))[0] == 1 else '안경 착용')


# [로지스틱 분류분석 문제3] 
#
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
#



  # Daily Time Spent on Site : 사이트 이용 시간 (분)

  # Age : 나이,

  # Area Income : 지역 소독,

  # Daily Internet Usage:일별 인터넷 사용량(분),

  # Clicked Ad : 광고 클릭 여부 (0, 1) 
  
  # 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
#
# ROC 커브와 AUC 출력

# 독립변수 : Daily Time Spent on Site, Age, Area Income, Daily Internet use
# 종속변수 : Clicked Ad

data3 =pd.read_csv("advertising.csv")
print(data3)
# print(data3.columns)
# Index(['Daily Time Spent on Site', 'Age', 'Area Income',
#        'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country',
#        'Timestamp', 'Clicked on Ad'],

x = data3[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]
y = data3['Clicked on Ad']
# print(x)
# print(y)

model =LogisticRegression().fit(x,y)
y_hat = model.predict(x)
print('y_hat(분류결과) : ',y_hat[:10])
print('실제값: ', y[:10])

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,y_hat))
# [[464  36]
#  [ 67 433]]
recall = 464/(464 + 36)        # 재현율 TPR
fallout = 67 / (67 + 433)      # 위양성율 FPR


from sklearn import metrics
cl_rep = metrics.classification_report(y,y_hat)
print('cl_rep : ',cl_rep)


f_valu = model.decision_function(x)
print('f_valu : ', f_valu)
fpr, tpr, thresholds = metrics.roc_curve(y,f_valu)
print('fpr : ', fpr)
print('tpr : ', tpr)
print('분류임계값  : ', thresholds)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,'o-', label='Logistic Regression')
plt.plot([0,1],[0,1], 'k--', label='classifier line(AUC:0.5')
plt.plot([fallout],[recall],'ro', ms=10) # 위양성율 , 재현율 값
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

# AUC ( Area Under the Curve ) : ROC 커브 면적
print('AUC : ',metrics.auc(fpr,tpr)) # 1에 가까울 수록 좋은 모델임
# AUC :  0.9580599999999999


