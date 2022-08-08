# 분류모델 성능 평가용 : ROC curve 그리기

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
x, y = make_classification(n_samples = 100, n_features = 2, n_redundant=0,random_state=123) 
#n_features 독립변수
print(x[:3], x.shape)
print(y[:3], y.shape)

# 산포도
import matplotlib.pyplot as plt
# plt.scatter(x[:,0],x[:,1])
# plt.show()

model =LogisticRegression().fit(x,y)
y_hat = model.predict(x)
print('y_hat(분류결과) : ',y_hat[:10])
print('실제값: ', y[:10])

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,y_hat))
# [[44  4]
#  [ 8 44]]

acc = (44 + 44)/ 100        # 정확도
recall = 44/(44 + 4)        # 재현율 TPR
precision = 44 / (44 +8 )   # 정밀도 (특이도, specificity)
fallout = 8 / (8 + 44)      # 위양성율 FPR
print('acc : ',acc)
print('recall : ',recall)
print('precision : ',precision)
print('fallout : ',fallout)
# 일반적으로 TPR은 1에 가까울 수록 좋고 , FPR은 0에 가까울 수록 좋다.
print()

from sklearn import metrics
ac_sco = metrics.accuracy_score(y,y_hat)
print('ac_sco : ',ac_sco)
cl_rep = metrics.classification_report(y,y_hat)
print('cl_rep : ',cl_rep)

# cl_rep :                precision    recall  f1-score   support
#
#            0       0.85      0.92      0.88        48
#            1       0.92      0.85      0.88        52
#
#     accuracy                           0.88       100
#    macro avg       0.88      0.88      0.88       100
# weighted avg       0.88      0.88      0.88       100

print()
# 판별함수(결정 함수, 불확실성 추정함수 )
f_valu = model.decision_function(x)
print('f_valu : ', f_valu)
fpr, tpr, thresholds = metrics.roc_curve(y,model.decision_function(x))
print('fpr : ', fpr)
print('tpr : ', tpr)
print('분류임계값  : ', thresholds)

# 시각화
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

