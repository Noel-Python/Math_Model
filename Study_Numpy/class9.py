#numpy广播 Broadcast

import numpy as np

a=np.array([1,2,3])
b=a
print(a*b)

a=np.array([[0],[10],[20],[30]])
a=np.tile(a,[1,3])
b=[1,2,3]
print(a+b)
