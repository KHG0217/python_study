# LogisticRegression 클래스 사용
# Pima 인디언 관련 당뇨병 데이터를 보고 당뇨병을 예측하는 분류 모델

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib, pickle

names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','age','Outcome']
df = pandas.read_csv('../testdata/pima-indians-diabetes.data.csv', header=None, names=names)
print(df.head(3),df.shape) # (768, 9)

arr = df.values
print(arr[:3])
x= arr[:, 0:8] # 메트릭스
y = arr[:, 8] # 벡터
print(x.shape)
print(y.shape)
print(set(y))

# train / test split

x_train, x_test, y_train, y_test =model_selection.train_test_split(x,y, test_size=0.3, random_state=7)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# 모델 생성
model = LogisticRegression()
model.fit(x_train, y_train)
print('예측값 : ', model.predict(x_test[:10]))
print('실제값 : ', y_test[:10])
print('예측값과 실제값이 다른 경우 갯수의 합 : ', (model.predict(x_test) != y_test).sum())

print('분류 정확도 : ', model.score(x_train, y_train)) # train 으로 정확도 확인 0.7839
from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
print('분류 정확도 : ',accuracy_score(y_test,pred)) # test 으로 정확도 확인 0.7489
# test와 train의 차이가 심하면  오버피팅 되있는것

# 학습된 모델 저장
joblib.dump(model, 'pima_model.sav')
# pickle.dump(model, open('pima_model.sav','wb'))
del model

# mymodel1 = pickle.load(open(joblib.load('pima_model.sav'))


# 학습된 모델 읽기
mymodel = joblib.load('pima_model.sav')
mypred = mymodel.predict(x_test)
print('분류 정확도: ', accuracy_score(y_test, mypred))

# 예측
print(x_test[:1])
new_data =[[ 1,90,62,12,43,27.2,0.58, 24]]
print('새로운 값 예측 결과 : ',mymodel.predict(new_data))
