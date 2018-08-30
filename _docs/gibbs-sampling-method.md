---
title: 吉布斯采样方法
permalink: /docs/probability/GibbsSamplingMethod/
excerpt: gibbs sampling method
created: 2017-06-02 18:50:29 +0200
---

## 概述

本文介绍Gibbs采样方法, 处理高维数据的采样问题. 在处理的过程中,高维数据的联合分布已知但不方便直接采样,而相应的条件分布比较好采样.那么可以根据条件分布采样来得到符合联合分布的多维数据的样本.

### 二维的Gibbs采样

回顾细致平稳条件:

$$ \pi(i)P(i,j) = \pi(j)P(j,i) $$

在二维的情况下上式中状态i或j,都变成一个二维的向量,记为 $(x_1,x_2)$. 注意上式中P(i,j)不是联合概率分布,正确的表述应该是条件概率分布. 对于联合概率分布一次采样得到n维向量,往往很难. Gibbs采样的思路为1次采样1维数据.

令

$$\pi(i) = \pi(x_1^t,x_2^t)$$
$$\pi(j) = \pi(x_1^t,x_2^{t+1})$$
$$P(j \vert i) = P(x_1^t,x_2^{t+1} \vert x_1^t, x_2^t) = \frac{P(x_1^t, x_2^t, x_2^{t+1})}{P(x_1^t,x_2^t)}$$  
$$P(i \vert j) = P(x_1^t,x_2^t \vert x_1^t, x_2^{t+1}) = \frac{P(x_1^t, x_2^t, x_2^{t+1})}{P(x_1^t,x_2^{t+1})} $$

代入细致平稳条件得到

$$ \pi(x_1^t,x_2^t) \frac{P(x_1^t, x_2^t, x_2^{t+1})}{P(x_1^t,x_2^t)} = \pi(x_1^t,x_2^{t+1}) \frac{P(x_1^t, x_2^t, x_2^{t+1})}{P(x_1^t,x_2^{t+1})} $$

化简得到

$$\pi(x_1^t,x_2^t) {P(x_1^t,x_2^{t+1})} = \pi(x_1^t,x_2^{t+1}) {P(x_1^t,x_2^t)} $$
$$\pi(x_1^t,x_2^t) P(x_1^t)P(x_2^{t+1} \vert x_1^t) = \pi(x_1^t,x_2^{t+1}) P(x_1^t)P(x_2^{t} \vert x_1^t)$$
$$\pi(x_1^t,x_2^t) P(x_1^t) P(x_2^{t+1} \vert x_1^t) = \pi(x_1^t,x_2^{t+1}) P(x-1^t) P(x_2^{t} \vert x_1^t)$$

最后得到,

$$\pi(x_1^t,x_2^t) P(x_2^{t+1} \vert x_1^t) = \pi(x_1^t,x_2^{t+1}) P(x_2^{t} \vert x_1^t)$$

上式就是一次采样一维变量的细致平稳条件.

$\pi$与P是一致的.也就是说 $\pi(x_1^t, x_2^t) = P(x_1^t,x_2^t) = P(x_1^t) P(x_2^t \vert x_1^t)$ .

细致平稳条件就可以写成

$$  P(x_1^t) P(x_2^t \vert x_1^t) P(x_2^{t+1} \vert x_1^t) = P(x_1^t) P(x_2^{t+1} \vert x_1^t) P(x_2^t \vert x_1^t) $$

很好,上式左右两边恒等. 结论为 $\pi$ 与P一致时满足细致平稳条件. 无需再考虑接受率的问题.


具体的操作步骤为

1. 随机初始化初始状态值 $x_1^1,x_2^1$
2. 从条件概率分布 $P(x_2|x_1^t)$ 中采样得到样本 $x_2^{t+1}$
3. 从条件概率分布 $P(x_1^{t+1} \vert x_2^{t+1})$ 中采样得到样本 $x_1^{t+1}$
4. 重复2,3步得到足够的样本 { $(x_1^1,x_2^1),(x_1^2,x_2^2),...,(x_1^t,x_2^t)$ }

### 二维的Gibbs采样例子

从二维正态分布$Norm(\mu,\Sigma)$,用Gibbs采样方法进行采样.

均值 $\mu = (\mu_1,\mu_2) = (5,-1)$  

协方差矩阵 $\Sigma = \left( \begin{array}{ccc} \sigma_X^2&\rho\sigma_X\sigma_Y \\  \rho\sigma_X\sigma_Y &\sigma_Y^2 \end{array} \right) =  \left( \begin{array}{ccc} 1&1 \\  1&4 \end{array} \right)$

联合概率密度函数为

$$ p(X,Y) = \frac{1}{2 \pi \sigma_X \sigma_Y \sqrt{1-\rho^2} } e^{-\frac{1}{2(1-\rho^2)} [\frac{(X-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(X-\mu_X)(Y-\mu_Y)}{\sigma_X\sigma_Y}+\frac{(Y-\mu_Y)^2}{\sigma_Y^2}]} $$

$\rho$ 为相关系数

$$ \rho_{X,Y} = \frac{cov(X,Y)}{\sigma_x \sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}$$

在例子中相应的就有

$$ \sigma_X = 1, \sigma_Y=2, \rho=0.5 $$



## 参考

刘建平的博客 http://www.cnblogs.com/pinard/p/6645766.html
