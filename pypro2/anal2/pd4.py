import pandas as pd
import numpy as np

# DataFrame 객체 병합 : merge 
df1 = pd.DataFrame({'data1':range(7), 'key':['b','b','a','c','a','a','b']})
print(df1)
df2 = pd.DataFrame({'key':['a','b','d'],'data2':range(3)})
print(df2)
print('inner ------')

print(pd.merge(df1, df2, on ='key')) # merge(기준치,기준치, on=기준값)/ key 를 기준으로 병합 (inner join : 교집합)
# key 중에서 공통으로 가지고 있는 'a' 와 'b'만 가져온다.
print()

print(pd.merge(df1, df2, on ='key', how ='inner' )) # how 병합방법 / 기본값 inner

print('outer ------')
print(pd.merge(df1, df2, on ='key', how ='outer' )) # key 를 기준으로 병합 (full outer join)
# key 값이 다나옴 'a','b','c','d'

print('left ------')
print(pd.merge(df1, df2, on ='key', how ='left' )) # key 를 기준으로 병합 (left outer join)
# left 여서 df1값은 다나옴 'a','b','c'

print('right ------')
print(pd.merge(df1, df2, on ='key', how ='right' )) # key 를 기준으로 병합 (right outer join)
# right 여서 df2값은 다나옴 'a','b','d'

print('공통 칼러명이 없는 경우 -------------')
df3 = pd.DataFrame({'key2':['a','b','d'],'data2':range(3)})
print(df3)
print(df1)
# df3와 df1은 칼럼명이 같지않음 .
print()

print(pd.merge(df1, df3, left_on ='key',right_on ='key2', how ='inner' ))
# 왼쪽은 key를 오른쪽은 key2로 inner 조인

print('자료 이어 붙이기') 
print(pd.concat([df1,df3], axis=0)) # 조인값 없이 그냥 이어 붙이기 행으로 ?
print()
print(pd.concat([df1,df3], axis=1)) # 조인값 없이 그냥 이어 붙이기 열로 ?

print('피봇(pivot) ------------------')
# 열을 기준으로 구조를 변경하여 새로운 집계표를 작성
data = {'city':['강남','강북','강남','강북'],
        'year':[2000,2001,2002,2002],
        'pop':[3.3,2.5,3.0,2.0]}
df = pd.DataFrame(data)
print(df)

print('privot------------------')
print(df.pivot('city', 'year', 'pop'))
# city가 행, year가 칼럼, pop가 벨류
print()

print(df.set_index(['city','year']).unstack())
# set_index : 기존의 행 인덱스를 제거하고, 첫번째 열 인덱스를 설정
# pivot과 같은 형태가 됨

print('groupby------------------') 
hap = df.groupby(['city'])
print(hap.sum())
print(df.groupby(['city']).sum()) # 위 두줄을 한 줄로 표현

print(df.groupby(['city','year']).mean()) # city 별 year별 pop 평균

print()
print(df.groupby(['city']).agg('sum'))

print()
print(df.groupby(['city','year']).agg('sum'))

print()
print(df.groupby(['city','year']).agg('mean'))

print()
print(df.groupby(['city','year']).agg(['mean','std']))

print('pivot _table-------------')
print(df)

print(df.pivot_table(index=['city'])) # 평균 계산 / 평균이 기본

print(df.pivot_table(index=['city'], aggfunc=np.mean)) # 위와 같은 값 # 결과는 year,pop값

print(df.pivot_table(index=['city','year'], aggfunc=[len,np.sum])) # 갯수와 합 #결과는 pop값

print(df.pivot_table(values=['pop'],index=['city'])) # city별 pop의 평균
print(df.pivot_table(values=['pop'],index=['city'], aggfunc=np.mean))
print(df.pivot_table(values=['pop'],index=['city'], aggfunc=len))
print(df.pivot_table(values=['pop'],index=['year'], columns=['city']))
print(df.pivot_table(values=['pop'],index=['year'], columns=['city'],
                     margins=True))
print(df.pivot_table(values=['pop'],index=['year'], columns=['city'],
                     margins=True, fill_value=0))
















