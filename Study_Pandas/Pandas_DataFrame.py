from sys import flags
import numpy as np
import pandas as pd

#pandas.DataFrame(data,index,columns,dtype,copy)
# data：一组数据(ndarray、series, map, lists, dict 等类型)。
# index：索引值，或者可以称为行标签。
# columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
# dtype：数据类型。
# copy：拷贝数据，默认为 False。

data=[['jack',12],['rose','11'],['frank','10']]
df=pd.DataFrame(data,columns=['name','age'],dtype=float)
print(df)

data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}
df = pd.DataFrame(data)
print (df)

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print (df)

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)
print('原始数据','\n',df)
print('第0行','\n',df.loc[0])
print('第1行','\n',df.loc[1])
print('第0 1行','\n',df.loc[[0,1]])

df=pd.DataFrame(data,index=['day1','day2','day3'])
print('with labels','\n',df)
print('label day2','\n',df.loc['day2'])
