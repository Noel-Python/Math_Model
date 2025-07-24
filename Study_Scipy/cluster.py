from scipy.cluster import vq
import numpy as np
import matplotlib.pyplot as plt

#标准化（按列）
data=np.array([[1.9,2.3,1.7],[1.5,2.5,2.2],[0.8,0.6,1.7]])
whiten_data=vq.whiten(data)
print(whiten_data)

#
code_book=np.array([[1.,1.,1.],[2.,2.,2.]])
features=np.array([[1.9,2.3,1.7],[1.5,2.5,2.2],[0.8,0.6,1.7]])
a=vq.vq(features,code_book)
print(a)

# kmeans聚类
# scipy.cluster.vq.kmeans(obs,k_or_guess,iter=20,thresh=1e-05,check_finite=True,*,seed=None)
# obs:m*n的数组，m为观测值个数，n为观测指标个数，每行为一点的坐标，必须先用whiten处理
# k_or_guess:传入k（int）时，为聚类的个数，传入guess（数组）时，为随机选择的初始质心
# iter:运行kmeans的次数，返回误差最小的结果，如果k_or_guess传入初始点坐标时，此参数可忽略
# thresh:终止阈值
# check_finite:设定输入是否包含有限数

# return
# 1.聚类中心点坐标
# 2.观测值与生成质心间的平均欧式距离（非平方）
features=np.array([[1.9,2.3],
                   [1.5,2.5],
                   [0.8,0.6],
                   [0.4,1.8],
                   [0.1,0.1],
                   [0.2,1.8],
                   [2.0,0.5],
                   [0.3,1.5],
                   [1.0,1.0]])
whiten=vq.whiten(features)
book=np.array((whiten[0],whiten[2]))
print('book','\n',vq.kmeans(whiten,book))

code=3
print('code','\n',vq.kmeans(whiten,code))

# Create 50 datapoints in two clusters a and b
pts = 50
rng = np.random.default_rng()
a = rng.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
b = rng.multivariate_normal([30, 10],
                            [[10, 2], [2, 1]],
                            size=pts)

features = np.concatenate((a, b))
# Whiten data
whitened = vq.whiten(features)
# Find 2 clusters in the data
codebook, distortion = vq.kmeans(whitened, 2)
# Plot whitened data and cluster centers in red
plt.scatter(whitened[:, 0], whitened[:, 1])
plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.show()

