#
#
# from io import StringIO
#
# from daal4py._daal4py import np
# from daal4py.sklearn.linear_model.logistic_path import LogisticRegression
# from sklearn.metrics._classification import accuracy_score
# from sklearn.model_selection._split import train_test_split
#
# import pandas as pd


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

# data3 =pd.read_csv("advertising.csv")
# print(data3)
# print(data3.columns)
# Index(['Daily Time Spent on Site', 'Age', 'Area Income',
#        'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country',
#        'Timestamp', 'Clicked on Ad'],

# x = data3[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]
# y = data3['Clicked on Ad']
# # print(x)
# # print(y)
#
# model =LogisticRegression().fit(x,y)
# y_hat = model.predict(x)
# print('y_hat(분류결과) : ',y_hat[:10])
# print('실제값: ', y[:10])
#
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y,y_hat))
# # [[464  36]
# #  [ 67 433]]
# recall = 464/(464 + 36)        # 재현율 TPR
# fallout = 67 / (67 + 433)      # 위양성율 FPR
#
#
# from sklearn import metrics
# cl_rep = metrics.classification_report(y,y_hat)
# print('cl_rep : ',cl_rep)
#
#
# f_valu = model.decision_function(x)
# print('f_valu : ', f_valu)
# fpr, tpr, thresholds = metrics.roc_curve(y,f_valu)
# print('fpr : ', fpr)
# print('tpr : ', tpr)
# print('분류임계값  : ', thresholds)
#
# import matplotlib.pyplot as plt
# plt.plot(fpr,tpr,'o-', label='Logistic Regression')
# plt.plot([0,1],[0,1], 'k--', label='classifier line(AUC:0.5')
# plt.plot([fallout],[recall],'ro', ms=10) # 위양성율 , 재현율 값
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('ROC curve')
# plt.legend()
# plt.show()
#
# # AUC ( Area Under the Curve ) : ROC 커브 면적
# print('AUC : ',metrics.auc(fpr,tpr)) # 1에 가까울 수록 좋은 모델임
# # AUC :  0.9580599999999999

# [Randomforest 문제3] 
#
# https://www.kaggle.com/c/bike-sharing-demand/data 에서  
# train.csv를 다운받아 bike_dataset.csv 으로 파일명을 변경한다.
# 이 데이터는 2011년 1월 ~ 2012년 12월 까지 
# 날짜/시간. 기온, 습도, 풍속 등의 정보를 바탕으로 1시간 간격의 자전거 대여횟수가 기록되어 있다. 
# train / test로 분류 한 후 대여횟수에 중요도가 높은 칼럼을 판단하여 
# feature를 선택한 후, 대여횟수에 대한 회귀예측을 하시오.  

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# from statsmodels.formula.api import ols
#
# import pandas as pd


# df = pd.read_csv("bike_dataset.csv")
# # print(df.head(1))
# # print(df.columns)
# # print(df.info())
# # print(df.isnull().sum())
#
# # 상관계수 확인
# print(df.corr('count'))
#
# # 칼럼지정
# df_x=df[['temp','atemp']]
# # print(df_x.columns)
# # print(df_x)
#
# df_y=df['count']
# # print(df_y)
#
# # train/test 분류
# x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.2,random_state=1 )
# # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# # (8708, 3) (2178, 3) (8708,) (2178,)
#
# # 모델
# rfmodel = RandomForestRegressor(n_estimators=1000,criterion='squared_error').fit(x_train,y_train )
# rfpredict = rfmodel.predict(x_test)
#
# # 예측값, 실제값
# print('예측값 : ', rfpredict[:10])
# print('실제값 : ', np.array(y_test)[:10])
#
# # 결점계수
# print('결점계수 : ', r2_score(y_test,rfpredict)) # 실제값,예측값
# # 결점계수 :  0.21158056015042548
#
# # 예측
# temp=int(input('temp(기온)  입력: '))
# atemp=int(input('atemp(체감온도) 입력: ')) 
#
#
# tdata = pd.Series(temp)
# adata = pd.Series(atemp)
# frame= pd.DataFrame()
#
# frame['temp']=tdata
# frame['atemp']=adata
# y_pred = rfmodel.predict(frame.values)
# print('cont(총대여량) ',y_pred)
# [XGBoost 문제] 
#
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
#
# RI    Na    Mg    Al    Si    K    Ca    Ba    Fe   
# import xgboost as xgb
#
# df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/glass.csv")
#
# print(df.head(3))
# # print(df.Type.unique()) # [1 2 3 5 6 7]
# x_feature =df.drop(['Type'], axis='columns')
# # print(x_feature.head(3))
# y_label= df['Type']
# print(y_label.head(3))
#
# # train/test 분류
# x_train, x_test, y_train, y_test = train_test_split(x_feature,y_label, test_size=0.2, random_state=12)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)
#
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#
# # # 모델생성
# model = xgb.XGBClassifier(booster='gbtree', max_depth=4, n_estimators=500).fit(x_train,y_train)
#
# # 예측
# pred = model.predict(x_test)
# print('예측값 :', pred[:10])
# print('실제값 : ', y_test[:10]) 
#
# from sklearn import metrics
# acc = metrics.accuracy_score(y_test, pred)
# print('정확도 : ',acc)
# # 정확도 :  0.023255813953488372
# print()
#
# from xgboost import plot_importance
# import matplotlib.pyplot as plt
# # 시각화 XGBClassifier 에서만 사용 가능
# fig, ax = plt.subplots(figsize=(10,12)) # 칼럼을 f숫자 으로 출력
# plot_importance(model,ax = ax)
# plt.show()




# [SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
# https://www.kaggle.com/zhaoyingzhu/heartcsv
# https://github.com/pykwon/python/tree/master/testdata_utf8         Heartcsv
#
#
# Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
# 각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
# dataset에 대해 학습을 위한 train과 test로 구분하고 
# 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
# 임의의 값을 넣어 분류 결과를 확인하시오.     
# 정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ
#
#
#
# feature 칼럼 : 문자 데이터 칼럼은 제외
# label 칼럼 : AHD(중증 심장질환)
#
#
#
# 데이터 예)
# "","Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"
#
# "1",63,1,"typical",145,233,1,2,150,0,2.3,3,0,"fixed","No"
#
# "2",67,1,"asymptomatic",160,286,0,2,108,1,1.5,2,3,"normal","s


# df=pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Heart.csv")
# # print(df.columns)
# print(df.head(3))
# print(df.isnull().sum())
# # Ca            4
# # Thal          2
#
# # nan 값 평균으로 대체 Ca
# df.loc[df['Ca'] != df['Ca'], 'Ca'] = df['Ca'].mean()
#
# # nan 값 제거 Thal
# df=df.dropna()
# print(df.isnull().sum())
#
# # # 칼럼지정
# df_x=df.iloc[:,1:-1]
# # print(df_x.columns)
# # print(df_x.head(3))
#
#
#
# df_y=df['AHD']
#
#
# # label을 dummy화
# # print(df_x['ChestPain'].unique()) # ['typical' 'asymptomatic' 'nonanginal' 'nontypical']
# # print(df_x['Thal'].unique()) #['fixed' 'normal' 'reversable' nan]
# df_x['ChestPain'] = df_x['ChestPain'].map({'typical':0,'asymptomatic':1,'nonanginal':2,'nontypical':3})
# df_x['Thal'] = df_x['Thal'].map({'fixed':0,'normal':1,'reversable':2})
#
# # print(df_y.unique()) #['No' 'Yes']
# df_y = df_y.map({'No':0,'Yes':1})
# # print(df_y[:3])
#
# # train/test 분류
# x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.2, random_state=12)
# # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# #(242, 13) (61, 13) (242,) (61,) 13개의 독립변수
#
# # model
# from sklearn import svm, metrics
# model = svm.SVC(C= 0.1).fit(x_train,y_train)
#
# # 예측한 결과
# pred = model.predict(x_test)
# print('예측값 : ', pred[:10])
# print('실제값 : ', y_test[:10].values)
#
# # 예측값 :  [0 0 0 0 0 0 0 0 0 0]
# # 실제값 :  [1 1 0 0 1 0 1 1 1 0]
# ac_score = metrics.accuracy_score(y_test,pred)
# print(ac_score) #0.6
#
# print()
# # 교차 검증 모델
# from sklearn import model_selection
# cross_vali = model_selection.cross_val_score(model, df_x, df_y, cv =3)
# print('각각의 검증 정확도 : ',cross_vali)
# print('평균 검증 정확도 : ',cross_vali.mean())
#
# # 각각의 검증 정확도 :  [0.54545455 0.53535354 0.53535354]
# # 평균 검증 정확도 :  0.5387205387205388
#
# # # 예측
# # Index(['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 
# #        'Fbs', 'RestECG', 'MaxHR',
# #        'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']
# Age=int(input('Age 입력: '))
# Sex=int(input('Sex 입력: ')) 
# ChestPain=int(input('ChestPain 입력: (typical:0,asymptomatic:1,nonanginal:2,nontypical:3:)'))
# RestBP=int(input('RestBP 입력: ')) 
# Chol=int(input('Chol 입력: ')) 
# Fbs=int(input('Fbs 입력: ')) 
# RestECG=int(input('RestECG 입력: ')) 
# MaxHR=int(input('MaxHR 입력: ')) 
# ExAng=int(input('ExAng 입력: ')) 
# Oldpeak=int(input('Oldpeak 입력: ')) 
# Slope=int(input('Slope 입력: ')) 
# Ca=int(input('Ca 입력: ')) 
# Thal=int(input('Thal 입력: (fixed:0,normal:1,reversable:2)'))
#
#
# tdata1 = pd.Series(Age)
# tdata2 = pd.Series(Sex)
# tdata3 = pd.Series(ChestPain)
# tdata4 = pd.Series(RestBP)
# tdata5 = pd.Series(Chol)
# tdata6 = pd.Series(Fbs)
# tdata7 = pd.Series(RestECG)
# tdata8 = pd.Series(MaxHR)
# tdata9 = pd.Series(ExAng)
# tdata10 = pd.Series(Oldpeak)
# tdata11 = pd.Series(Slope)
# tdata12 = pd.Series(Ca)
# tdata13 = pd.Series(Thal)
#
#
# frame= pd.DataFrame()
#
# frame['Age']=tdata1
# frame['Sex']=tdata2
# frame['ChestPain']=tdata3
# frame['RestBP']=tdata4
# frame['Chol']=tdata5
# frame['Fbs']=tdata6
# frame['RestECG']=tdata7
# frame['MaxHR']=tdata8
# frame['ExAng']=tdata9
# frame['Oldpeak']=tdata10
# frame['Slope']=tdata11
# frame['Ca']=tdata12
# frame['Thal']=tdata13
#
# y_pred = model.predict(frame.values)
# print('AHD(중증 심장질환): ',y_p





#  testdata 폴더에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다.
# 만들어진 회귀문제 모델을 이용하여 아래의 문제를 해결하시오. (ols 함수를 사용하시오.)
#
# - 키보드로 국어 점수를 입력하면 수학 점수 예측 (80점을 입력하자)배점:
# import pandas as pd
#
# df=pd.read_csv("testdata/student.csv ")
#
# # print(df.head(3))
#
# df_x=df['국어']
# df_y=df['수학']
# from statsmodels.formula.api import ols
# model = ols('수학 ~ 국어', data=df).fit()
#
# pred = model.predict(df_x)
#
# x=[80]
#
# frame= pd.DataFrame()
# frame['국어']=x
#
# y_pred = model.predict(frame['국어'])
# print('수학점수 예측: ',y_pred.values)


# import pandas as pd
# x = 1,2,3,4,5
# y = 8,7,6,4,5
#
# df = pd.DataFrame({'x':x, 'y':y})
#
# print(df.corr())
#
# import pandas as pd
#
# data = pd.read_csv('testdata/titanic_data.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age','Fare'])
# print(data.head(2), data.shape) # (891, 12)
# data.loc[data["Sex"] == "male","Sex"] = 0
# data.loc[data["Sex"] == "female", "Sex"] = 1
# print(data["Sex"].head(2))
# print(data.columns)
#
# feature = data[["Pclass", "Sex", "Fare"]]
# label = data["Survived"]
#
# # 이하 소스 코드를 적으시오.
# # 1) train_test_split (7:3), random_state=12
# # 2) DecisionTreeClassifier 클래스를 사용해 분류 모델 작성
# # 3) 분류 정확도 출력 (배점:10)
#
# from sklearn.model_selection._split import train_test_split
# from sklearn.metrics import r2_score
# x_train, x_test, y_train, y_test =train_test_split(feature,label, test_size=0.3, random_state=12) 
# # 실습 1 : DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor(criterion='squared_error').fit(x_train,y_train)
# print('예측값 : ', model.predict(x_test)[:5])
# print('실제값 : ', y_test[:5])
# print('결점계수 : ', r2_score(y_test,model.predict(x_test))) #결점계수 :  0.1473600932288931
# print()
#


# feature(독립) columns :
# fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
# free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
# label(종속) column : quality (배점:10)
# 코드 시작
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("winequality-red.csv")
df_x = df.drop(['quality'], axis=1)  # feature로 사용. quality를 제외한 나머지 열
df_y = df['quality']  # label로 사용
print(df_y.unique())  # [5 6 7 4 8 3]
print(df_x.columns)  # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', ...

# 이하 소스 코드를 적으시오.
# 1) train_test_split (7:3), random_state=12
x_train, x_test, y_train, y_test =train_test_split(df_x,df_y, test_size=0.3, random_state=12) 

# 실습 2 : RandomForestClassifier
model = RandomForestClassifier(criterion='entropy', n_estimators=500).fit(x_train,y_train)
pred = model.predict(x_test)
print('예측값 : ', model.predict(x_test)[:5])
print('실제값 : ', y_test[:5])
print('분류 정확도 : ', accuracy_score(y_test,pred)) # 분류 정확도 :  0.6638




