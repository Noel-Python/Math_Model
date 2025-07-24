# numpy创建数组

import numpy as np
x=np.empty([3,2],dtype=int)
print(x)

x=np.zeros([3,4])
print(x)
x=np.zeros([2,3],dtype=int)
print(x)
x=np.zeros([3,2],dtype=[('x','i4'),('y','i4')])
print(x)

x=np.ones([3,3],dtype=int)
print(x)
