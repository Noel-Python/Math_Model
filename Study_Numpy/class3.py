# numpy 数组属性
import numpy as np
a=np.arange(24)
print(a.ndim)
b=a.reshape(2,4,3)
print(b.ndim)
print(a,'\n',b)

c=np.array([[1,2,3],[4,5,6]])
print(c.shape)

# 返回元素的字节大小
x=np.array([1,2,3,4,5],dtype=np.int8)
print(x.itemsize)

# 返回ndarray对象的内存信息
print(x.flags)
