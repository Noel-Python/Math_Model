from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris.data
tar = iris.target# target为分类标记
tar_name=iris.target_names
print(tar_name)
