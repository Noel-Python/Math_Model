# numpy 数组操作

import numpy as np
a=np.arange(9)
b=a.reshape(3,3)
print(b)

#np.ndarray.flat 数组元素迭代器
for i in b:
    print(i)
for i in b.flat:
    print(i)

# flatten 对拷贝的修改不会影响原数组
#order:'C'-按行(缺省值) 'F'-按列 'A'-原顺序 'K'-元素在内存中的出现顺序
a=np.arange(8).reshape(2,4)
print('原数组','\n',a)
print('展开','\n',a.flatten())
print('F展开','\n',a.flatten(order='F')) 
print(a)

# ravel 对拷贝的修改会影响原数组
a = np.arange(8).reshape(2,4)
print ('原数组：')
print (a)
print ('\n')
print ('调用 ravel 函数之后：')
print (a.ravel())
print ('\n')
print ('以 F 风格顺序调用 ravel 函数之后：')
print (a.ravel(order = 'F'))
print(a)

#转置
a=np.arange(12).reshape(3,4)
print(np.shape(np.transpose(a)))
print(np.shape(a.T))


