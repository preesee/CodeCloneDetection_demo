import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 使用datasets提供的方法构建非线性数据
cX, y = datasets.make_moons(noise=0.20)

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()
