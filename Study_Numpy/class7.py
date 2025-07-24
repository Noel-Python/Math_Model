#numpy 切片和索引

import numpy as np
a=np.arange(10)
print(a)
s=slice(2,7,2)
print(a[s])
s=a[2:7:2]
print(s)
print(a[5])
print(a[2:])
print(a[2:5])
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
a1=a[...,1]
a2=a[1,...]
a3=a[...,1:]
print(a1,'\n',a2,'\n',a3)
