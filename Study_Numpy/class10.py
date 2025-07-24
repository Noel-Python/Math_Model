#numpy迭代数组

from asyncore import readwrite
from re import X
import numpy as np

a=np.arange(6).reshape(2,3)
print('原始数据：','\n',a)
#np.nditer为迭代器
#order='F'/'C' 列/行优先 缺省时为行优先

a1=np.copy(a,order='C')
print('a1',a1)
for i in np.nditer(a1):
    print(i)
a2=np.copy(a,order='F')
print('a2',a2)
for i in np.nditer(a2):
    print(i)

print('行优先')
for i in np.nditer(a,order='F'):
    print(i)
print('列优先')
for i in np.nditer(a,order='C'):
    print(i)

#修改数组中元素的值
# op_flags1缺省时为read-only，指定为read-write后可修改
a=np.arange(0,60,5)
a=a.reshape(3,4)
print('原数组为：','\n',a)
for i in np.nditer(a,op_flags=['readwrite']):
    i[...]=2*i
print('修改后的数组为：','\n',a)
