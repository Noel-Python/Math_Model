# 清洗空值
from pickle import TRUE
import pandas as pd
demodata=pd.read_csv('property-data.csv')
print(demodata.to_string())
# 该数据中的空数据包括：n/a NA -- na

# 清洗空值
# DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。
# how：默认为 'any' 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 how='all' 一行（或列）都是 NA 才去掉这整行。
# thresh：设置需要多少非空值的数据才可以保留下来的。
# subset：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数。
# inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。

# isnull() 判断各个单元格是否为空
print(demodata['NUM_BEDROOMS'])
print(demodata['NUM_BEDROOMS'].isnull())

# 指定空数据类型
missing_value=['n/a','na','--']
demodata=pd.read_csv('property-data.csv',na_values=missing_value)
print(demodata['NUM_BEDROOMS'])
print(demodata['NUM_BEDROOMS'].isnull())

# 删除含有空数据的行
demodata=pd.read_csv('property-data.csv')
delete_data=demodata.dropna()
print(delete_data.to_string())

# 删除含有空数据的行（指定列中寻找空数据）
delete_data=demodata.dropna(subset=['PID'],inplace=True)
print(demodata.to_string())

# 替换空字段
missing_value=['n/a','na','--']
demodata=pd.read_csv('property-data.csv',na_values=missing_value)
a=demodata.fillna('filled')
print(a.to_string())

# 替换空字段（在指定列）
demodata['PID'].fillna(0.0,inplace=True)
print(demodata.to_string())

# mean() median() mode() 平均数 中位数 众数
demodata=pd.read_csv('property-data.csv',na_values=missing_value)
fillvalue_ST_NUM=demodata['ST_NUM'].mean()
# fillvalue_ST_NUM=demodata['ST_NUM'].median()
# fillvalue_ST_NUM=demodata['ST_NUM'].mode()
demodata['ST_NUM'].fillna(fillvalue_ST_NUM,inplace=True)
print(demodata.to_string())

# 格式化数据(日期)
data={"data":['2022/11/03','2022/03/31','20221108'],"age":[23,17,35]}
data=pd.DataFrame(data,index=['day1','day2','day3'])
print('data','\n',data.to_string())
data['data']=pd.to_datetime(data['data'])
print(data.to_string())

# 寻找重复数据
data={'name':['rose','doge','doge','jackson'],'age':[12,11,11,13]}
data=pd.DataFrame(data)
print(data.duplicated())
# 删除重复数据
data.drop_duplicates(inplace=True)
print(data.to_string())
