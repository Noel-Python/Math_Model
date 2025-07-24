#numpy 高级索引
import numpy as np

#整数数组索引
x=np.array([[1,2],[3,4],[5,6]])
y=x[[0,1,2],[0,1,0]]
print(y)

x=np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print('x为:\n',x)
y=x[[[0,0],[3,3]],[[0,2],[0,2]]]
print('y为:\n',y)

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=a[1:3,1:3]
c=a[1:3,[0,2]]
d=a[...,1:]
print('a',a)
print('b',b)
print('c',c)
print('d',d)

#布尔索引
print('x中大于5的元素',x[x>5])

a=np.array([np.nan,1,2,np.nan,3])
print(a[~np.isnan(a)])

a=np.array([1,2+1j,3,4,5+2j])
b=a[np.iscomplex(a)]
print(b)

#花式索引
x=np.arange(9)
x2=x[[0,6]]
print(x2)

x=np.arange(32).reshape(8,4)
print(x)
print(x[[4,2,1,7]])
print(x[[-4,-2,-1,-7]])

a=[1,2]
b=[3,4]
print(x[np.ix_([1,5,7,2],[0,3,1,2])])
