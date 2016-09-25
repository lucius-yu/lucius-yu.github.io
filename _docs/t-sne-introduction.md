---
title: t-sne简介
permalink: /docs/machine learning/t-SNE/
excerpt: t-sne introduction
created: 2016-09-14 11:28:26 +0200
---

## 概述
t-SNE是一种非线性数据降维方法，对高维数据降维到2维或者3维有助于将数据用图像的方式展示出来. 该方法是SNE方法的改进方法, 关于SNE的介绍请参考我写的<<Stochastic Neighbor Embedding非线性降维简介>>

主要的改进有两点

1. 高维数据点的两两相似(相邻)的分布由非对称改成对称形式
2. 低维数据点的两两相似(相邻)的分布有Gaussian核改成t分布

##  高维数据点的相邻概率分布改进

SNE中的定义为

$$ p(j \vert i)=\frac{exp(-\frac{ \Vert x_i - x_j \Vert ^2} {2\sigma^2_i})}{\sum_{k \neq i}exp(-\frac{ \Vert x_i - x_k \Vert ^2} {2\sigma^2_i})} $$

首先上式中的概率分布本质上是条件概率分布,所以改写成$p(j \vert i)$
其次上式是非对称的, $\sigma_i$不一定等于$\sigma_j$, 分母中$x_i$到$x_k$的距离也不等于$x_j$到$x_k$的距离.

如果我们需要修改成对称形式可以简单的修改为
$$ p(j \vert i)=\frac{exp(-\frac{ \Vert x_i - x_j \Vert ^2} {2\sigma^2})}{\sum_{k \neq l}exp(-\frac{ \Vert x_k - x_l \Vert ^2} {2\sigma^2})} $$

* 将$\sigma$固定,不随点的改变而改变
* 分母也不是用点i到其他点的距离,而是用所有点的两两距离

但是上面的修改并不好, 实际是定义一个对称的联合概率为对称的条件概率, 式子如下
$$p_{i,j} = \frac{p(j \vert i)+p(i \vert j)}{2n}$$   
n为所有点的数量

首先修改后仍然可以用KL散度做cost function进行梯度下降.  

这样修改的原因
1. 修改成联合概率后会让后面的cost function的梯度下降更简单一点
2. 如果在高维分布中如果有些离群点,这些点到到其他点都较远,这些离群点在低维分布中就很难定,改成这种形式后能保证$p_{i}=\sum_j{p_{i,j}}>\frac{1}{2n}$

KL散度用以衡量两个分布的相似度,当用梯度下降不断调正低维空间点的位置来最小化KL散度时,实际上是将低维空间点的分布逼近高维空间的分布

$$ C=KL(P \Vert Q)=\sum_i\sum_j{p_{i,j}log\frac{p_{i,j}}{q_{i,j}}} $$

对低维空间点i的偏导如下

$$\frac{\partial{C}}{\partial{y_i}} = 4\sum_j(p_{i,j}-q_{i,j})(y_i-y_j)$$

##  低维空间的概率分布改进


## 参考
1. [Visualizing Data using t-SNE](http://www.cs.toronto.edu/~hinton/absps/tsne.pdf), Laurens van der Maaten, Geoffrey Hinton
