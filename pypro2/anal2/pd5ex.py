import numpy as np
import pandas as pd

print('3-1')
data =  pd.read_csv('testdata/titanic_data.csv')

fr = pd.DataFrame(data)
# print(fr)
bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]

result_cut1 = pd.cut(fr['Age'], bins, labels = labels) 

fr2 =pd.DataFrame(pd.value_counts(result_cut1))
print(fr2)

print('3-2')
print(data.pivot_table(
    values=['Survived'],index=['Sex','Age'], columns=['Pclass']))


print('4-2')
data2 =  pd.read_csv('testdata/tips.csv')
print(data2.head(3))
print(data2.describe()) # 요약 통계량 보기

smoker_result =pd.DataFrame(pd.value_counts(data2['smoker']))
print(smoker_result)
print(data2["day"].unique())



