import numpy as np
import pandas as pd
import os

# CSV:Comma-Separated Values 字符分隔值

nbadata=pd.read_csv('nba.csv')
# 仅展示前后各5行
# print(nbadata)
print(nbadata.to_string())

name = ["Google", "Runoob", "Taobao", "Wiki"]
site = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
age = [90, 40, 80, 98]   
dict = {'name': name, 'site': site, 'age': age}
df = pd.DataFrame(dict)
# 保存 dataframe
df.to_csv('site.csv',index=0)
a=pd.read_csv('site.csv')
print(a.to_string())

# CSV数据读取
print('head10','\n',nbadata.head(10))
print('tail10','\n',nbadata.tail(10))

# CSV信息读取
print('nbadata_info','\n',nbadata.info())
