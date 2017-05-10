---
title: 主成分分析简介
permalink: /docs/machine learning/PCA/
excerpt: PCA introduction
created: 2017-03-17 22:50:15 +0200
---

## 概述

主成分分析在机器学习领域中主要用于数据降维. 由于本人是做无线通信的,在通信领域中波束赋形也就是
beamforming用的也是同样的技术. 本文结合来谈谈主成分分析

## 最大方差

考虑一组M个观测数据${x_m}$, 每个数据是一个n维的数据,目标是将数据投影到一个d维空间 d<n, 同时最大化投影数据的方差.

先看一个直观的图像, 下图为400个二维的数据样本.这二维的数据相关性很大,当数据投影到绿色的箭头的方向上时,方差最大

![Correlated Data]({{ site.url}}/doc-images/machine-learning/pca-01.PNG)

考虑将数据投射到1维的情况,也就是在n维的向量空间找到一个向量u,当数据集中的数据投射到该向量上时投射后的数据集具有最大方差,实际上只需要找出向量u的方向即可.所以可以约束向量u的模为1,于是数据投射到该向量上的操作就是数据$x_m$点乘向量u.

投射后的方差为  

$$ \frac{1}{M} \sum (u^T x_m - u^T x^-)^2 = u^T S u  $$

其中 $x^-$ 为数据集X的均值, $S=\frac{1}{M} \sum(x_m - x^-)(x_m - x^-)^T$ 为数据集X的协方差矩阵. 问题的目标就转换为在约束向量u为单位向量(即$u^Tu=1$)的情况下,最大化$u^T S u$, 采用拉格朗日乘子法将约束优化问题转为无约束优化得到下式

$$ u^T S u + \lambda(1-u^Tu) $$

对上式中u求偏导并设为0得到

$$ S u = \lambda u $$

$S$为nxn的协方差矩阵,$\lambda$为一个标量, $u$为n维向量.

回顾矩阵特征向量的定义, 可知$u$即为矩阵$S$的特征向量, $\lambda$为对应的特征值.

## 测试代码

```
import numpy as np
import matplotlib.pyplot as plt

## generate the data
M = 400
x=np.random.randn(2,M)
cov = [[0.95,0.36],[0.36,0.312]]
d = np.matmul(x.T,cov).T

## plot the data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(d[0],d[1])

## calculate eigen vector
w,v=np.linalg.eig(np.matmul(d,d.T) / M)

## plot the eigen vector
ax.arrow(0, 0, v[0,0], v[1,0], width=0.02,head_width=0.05, head_length=0.1, fc='g', ec='g')

plt.show()
```

## 其他

也可以从最小误差的角度看pca的问题.

## 参考

PRML
