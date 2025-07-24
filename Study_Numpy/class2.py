# numpy 数据类型
import numpy as np
dt=np.dtype(np.int32)
print(dt)

# int8,int16,int32,int64 四种数据类型可以使用字符串'i1','i2,'i4','i8'代替
dt1=np.dtype('i4')
print(dt1)

#创建结构化数据类型
dt2=np.dtype([('age',np.int8)])
# 将数据类型应用于ndarray对象
a=np.array([(10,),(20,),(30,)],dtype = dt2)
print(a)

student=np.dtype([('name','S20'),('age','i1'),('marks','f4')])
print(student)
student1=np.array([('abc',21,50),('def',18,75)],dtype=student)
print(student1)
