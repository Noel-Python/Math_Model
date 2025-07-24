#numpy从数值范围创建数组
import numpy as np 

#np.arange(start,end,step,dtype)
x=np.arange(5)
print(x)
x=np.arange(5,dtype=float)
print(x)
x=np.arange(10,20,2)
print(x)

#np.linspace(start,stop,num=50,endpoint=True,retstep=False,dtype=None)
a=np.linspace(1,10,10)
print(a)
a=np.linspace(10,20,5,endpoint=False)
print(a)
a=np.linspace(1,10,10,retstep=True)
print(a)

#等比数列
#np.logspace(start,stop,num=50,endpoint=True,base=10.0,dtype=None)
a=np.logspace(1.0,2.0,num=20)
print(a)
a=np.logspace(0,9,10,base=2)
print(a)
