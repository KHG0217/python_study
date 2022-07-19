import pandas as pd
import numpy as np


# DataFrame 구조 변경 : stack, unstack

df = pd.DataFrame(1000 + np.arange(6).reshape(2,3),
                  index=['대전','서울'], columns=['2020', '2021', '2022'])
print(df)
print()

df_row = df.stack() # 열을 행으로 변경하는 함수,  인덱스를 기준으로 칼럼쌓기 [그룹화]
print(df_row)
print()

df_col = df_row.unstack() # stack 결과를 해제한다.
print(df_col)

print('연속형 자료 범주화(구간 설정)')

price = [10.3, 5.5, 7.8, 3.6]
cut = [3, 7, 9, 11] # 구간 기준값
result_cut = pd.cut(price, cut)

print(result_cut) # (a,b] -> a < x <=b (a초과 b이하)

print(pd.value_counts(result_cut)) # 구간별 값갯수 보기

print(type(result_cut)) # <class 'pandas.core.arrays.categorical.Categorical'>
print()

datas = pd.Series(np.arange(1,1001))
print(datas.head(3))
print(datas.tail(3))

result_cut2 = pd.qcut(datas, 3) # datas 값을 3개로 나눈다
print(result_cut2)
print(print(pd.value_counts(result_cut2)))
print()

group_col = datas.groupby(result_cut2)
print(group_col.agg(['count', 'mean', 'std', 'min'])) # agg(['함수명'] 써놓은 함수 그룹에 대해 함수를 실행함

# agg 대신 함수를 직접 작성하기
def summary_func(gr):
    return {'count':gr.count(), 'mean':gr.mean(), 'std':gr.std(), 'min':gr.min()}

print(group_col.apply(summary_func)) # 열방향으로 구조가 짜여짐
print(group_col.apply(summary_func).unstack()) # unstack 함수로 행방향으로 풀기