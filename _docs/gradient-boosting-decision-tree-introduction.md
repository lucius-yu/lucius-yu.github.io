---
title: Gradient Boosting Decision Tree 简介
permalink: /docs/machine learning/gradient-boosting/
excerpt: gradient-boosting introduction
created: 2016-11-08 18:28:26 +0200
---

# 总体概览

本文材料基本选自台湾大学林轩田的机器学习技法课程.

* AdaBoost Decision Tree 回顾
* AdaBoost 的优化问题
* Gradient Boosting

## AdaBoost Decision Tree 回顾

### 基本算法

给定训练数据集$D$, 设一共建$T$棵树.

For t = 1, 2, ... T  
*  从数据集$D$中,对每个样本计算权重$u^{(t)}$
*  用加权后的样本数据来计算得到一个决策树$DTree(D,u^{(t)})$
*  对获得的决策树计算其投票权重

最后对所有获得的决策树进行加权线性合并.

### 样本权重的选择

* 样本的初始权重为均匀分布,每个样本权重都为$1/N$
* 定义缩放因子
$$ \diamond_t = \sqrt \frac{1-\epsilon}{\epsilon} $$
* 错误样本的权重乘以缩放因子
$$ u_n^{t+1} = u_n^t * \diamond_t $$
* 而正确样本的权重要除以缩放因子
$$ u_n^{t+1} = u_n^t / \diamond_t $$

### 线性合并的权重

对于某个弱分类器,这里也就是一个限制了高度的树,其在最后线性合并时的权重为

$$ \alpha_t = ln(\diamond_t) $$

## AdaBoost 的优化问题

### 代价函数(错误度量)的问题.

定义一个二分类问题第n个样本的真实输出标签为$y_n \in [-1,1]$, 在AdaBoost聚合时第t个分类器的输出为$g_t(x_n)$

可以将样本权重的两个更新公式直接合并为一个公式

$$ u_n^{t+1} = u_n^t \diamond_t^{-y_n g_t(x_n)} $$

更新权重的定义
$$ \alpha_t = ln(\diamond_t) $$

于是
$$ \diamond_t = exp(\alpha_t) $$

导出
$$ u_n^{t+1} = u_n^t exp^{-y_n \alpha_t g_t(x_n)} $$
$$ u_n^{t+1} = u_n^1 \prod_{t=1}^{T} exp^{-y_n \alpha_t g_t(x_n)} = \frac{1}{N}  exp^{-y_n \sum_{t=1}^{T} \alpha_t g_t(x_n)} $$

而$\sum_{t=1}^T \alpha_t g_t(x) $,就是直到t轮的合并结果,记为voting score,即投票分数.记为s

则第t+1轮样本点n的权重就为下式

$$u_n^{t+1} = \frac{1}{N}exp^{-y_n s_n}$$

回到一个基本的问题,我们需要什么样的$u_n^{t+1}$, 或者说我们要什么样的所有样本点权重之和$\sum_{n=1}^{N} u_n^{t}$. 是越来越大,还是越来越小. 很简单, 随着弱分类器的增加组合,分对样本会越来越多, Training Error也会越来越小, 我们要的这个样本点权重之和也要越来越小. 如果这个样本点权重之和停止减小, 也意味着Training Error会停止减小.

于是整个问题就成为一个最小化问题

$$argmin(\sum_{n=1}^N u_n^{T+1}) = argmin(\sum_{n=1}^N \frac{1}{N}exp^{-y_n \sum_{t=1}^T \alpha_t g_t(x_n)}) = argmin(\sum_{n=1}^N \frac{1}{N}exp^{-y_n s_n}) $$

如果我们新加入一个分类器$h$并给一个权重$\eta$ 下面的函数就可以作为代价函数在进行优化.

$$ min_h E_{ADA}= \sum_{n=1}^N \frac{1}{N}exp^{-y_n (\sum_{t=1}^T \alpha_t g_t(x_n) + \eta h(x_n))} $$

也就是说找到一个$h(x)$和一个$\eta$在最小化上面的error函数, 首先将上式简化为

$$ E_{ADA} = \sum_{n=1}^N u_n^t exp^{-y_n \eta h(x_n)} $$


### 最优新添加的函数

也就是寻找$h(x_n)$其能最小化$E_{ADA}$

对$E_{ADA}$在原点处,做一阶泰勒展开来近似

$$
\begin{aligned}
min_h E_{ADA} = \sum_{n=1}^N u_n^t exp^{-y_n \eta h(x_n)} \\  
          \approx \sum_{n=1}^N u_n^t (1 -y_n \eta h(x_n))\\
          = \sum_{n=1}^N u_n^t + \sum_{n=1}^N u_n^t (-y_n) \eta h(x_n)
\end{aligned}
$$

$\sum u_n^t$是由上一轮计算出的, 此时为常数,所以就只需要最小化$\eta \sum_{n=1}^N u_n^t (-y_n) h(x_n)$, 在优化$h(x_n)$时,$\eta$可以认为是常数, 此时最优的$h(x_n)$就应能最小化$\sum_{n=1}^N u_n^t (-y_n) h(x_n)$

对于二分类问题

$$
\begin{aligned}
\sum_{n=1}^N u_n^t (-y_n) h(x_n) = \sum_{n=1}^N u_n^t \{_{+1 \quad if \quad y_n \neq h(x_n)}^{-1 \quad if \quad y_n = h(x_n)} \\
= -\sum_{n=1}^N u_n^t + \sum_{n=1}^N u_n^t \{_{2 \quad if \quad y_n \neq h(x_n)}^{0 \quad if \quad y_n = h(x_n)} \\
= -\sum_{n=1}^N u_n^t + 2E_{in}^{u^t}
\end{aligned}
$$

最后, 要选择$h(x_n)$让上式最小化, 也就是选择$h(x_n)$让$E_{in}^{u^t}$最小. 换言之,就是新生成的树在$u^t$为样本权重时训练错误(training error)最小的树. 这个树在AdaBoost Decision Tree中就是第t轮生成的$g_t(x)$.

### 最优的新加树的权重

在找到最优的$h(x_n) = g_t(x_n)$之后,就要决定$\eta$能最小化我们的代价函数.

$$ min_\eta E_{ADA} = \sum_{n=1}^N u_n^t exp^{-y_n \eta g_t(x_n)} $$

对上面的代价函数分两种情况并重写得到下面的式子

$$
\begin{aligned}
y_n = g_t(x_n) : u^t exp^{-\eta} \\
y_n \neq g_t(x_n) : u^t exp^{\eta} \\
\Rightarrow E_{ADA} = (\sum_{n=1}^N u_n^t) ((1-\epsilon_t) exp(-\eta) + \epsilon_t exp(\eta))
\end{aligned}
$$

上式对$\eta$求偏导并设为0,可以解得$\eta$

$$
\begin{aligned}
\frac{\partial E_{ADA}}{\partial \eta} = 0 \\
\Rightarrow -(1-\epsilon_t) exp^{-\eta} + \epsilon_t exp^{\eta} = 0 \\
\Rightarrow \epsilon_t (exp^{\eta})^2 = 1 - \epsilon_t \\
\Rightarrow \eta = ln (\sqrt{\frac{1-\epsilon_t}{\epsilon_t}})
\end{aligned}
$$

最优的权重就是AdaBoost在t轮计算得到的$g_t(x_n)$的权重$\alpha_t = ln (\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}) $

### 小结

对AdaBoost的优化问题,只是从梯度下降优化的视角来看待AdaBoost在线性加权弱分类树这一做法的合理性和最优性.

小结一下, 一共三点

* 在AdaBoost中对应的代价函数(错误度量函数)就是所有样本点权重之和. 在t轮之后的残余错误越大,对应的下一轮的$\sum_{n=1}^N u_n^{t+1}$也就越大,目标就是不断优化减小这个所有样本点权重之和.
* 对代价函数做一阶泰勒展开后,证明第t轮最优的新添加树,就是在AdaBoost Decision Tree中, 根据第t轮样本权重$u_t$训练的树$g_t(x_n)$
* 用Steepest Gradient Descent,也就是求一阶偏导为0是的$\eta$,证明对t轮新加入的树的最优合并权重就是AdaBoost Decision Tree用的$\alpha_t = ln(\sqrt{\frac{1-\epsilon_t}{\epsilon_t}})$


## Gradient Boost Desicion Tree

在AdaBoost的优化问题中,给出了一个聚合模型的代价函数和用梯度下降的方式(Steepest Descent)来优化代价函数的方式. 实际上, 对于不同的问题,譬如回归问题,多类分类问题,可以将AdaBoost Decision Tree一般化来采用不同的代价函数, 并在每一轮用残差,(注意，这里使用残差), 来对新加入的弱分类树或弱回归树,用梯度下降的方法求合并权重. 可以用来解决回归，多类分类等等不同问题. 对于二分类实际上也可以用logistic regression的代价函数来解决(AdaBoost Descision Tree采用了一种特殊的代价函数). 用这种方式扩展后就形成了Gradient Boost Decision Tree. 实际上被聚合的也不一定是树. 也就是$Err$函数和$h(x_n)$都可以被一般化.


### Gradient Boosting 的回归优化问题

下面还是以弱回归树用Gradient Descent聚合的方式来看看Gradient Descent Descision Tree如何处理回归问题. 采用最小均方差为代价函数.

AdaBoost的优化问题

$$ min_\eta \quad min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N e^{(-y_n \sum_{\tau=1}^{t-1} \alpha_\tau g_\tau(x_n)+\eta g_t(x_n))} $$

Gradient Boost的优化问题

$$ min_\eta \quad min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N err(\sum_{\tau=1}^{t-1} \alpha_\tau g_\tau(x_n)+\eta g_t(x_n), y_n) $$

对于回归问题,我们用均方差为错误(代价函数)
$$err(s,y) = (s-y)^2$$

记$s_n = \sum_{\tau=1}^{t-1} \alpha_\tau g_\tau(x_n)$,即前t-1个弱回归器对第n个样本的线性加权后的估计值

### 新加入的回归器的优化问题

思路上有两个,结果没区别.　

#### 第一个思路

对Err函数做一阶泰勒展开
$$
\begin{aligned}
min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N err(\sum_{\tau=1}^{t-1} \alpha_\tau g_\tau(x_n) + \eta g_t(x_n), y_n) \\
\approx min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N err(s_n,y_n) + \frac{1}{N} \sum_{n=1}^N \eta g_t(x_n) \frac{\partial err(s,y_n)}{\partial s_n} \\
= min_{g_t} \quad constants + \frac{\eta}{N} \sum_{n=1}^N g_t(x_n) * 2(s_n - y_n)
\end{aligned}
$$

由于第一项是常数,实际问题就等价于
$$ min_{g_t} \quad \frac{\eta}{N} \sum_{n=1}^N g_t(x_n) * 2(s_n - y_n) $$

加入正则项Regularization,来约束$g_t$, 将优化问题改写为

$$
\begin{aligned}
min_{g_t} \quad \frac{\eta}{N} \sum_{n=1}^N (g_t(x_n) * 2(s_n - y_n) + g_t(x_n)^2) \\
= min_{g_t} \quad \frac{\eta}{N} \sum_{n=1}^N ( constant + (g_t(x_n) -  (y_n - s_n))^2 ) \\
\Rightarrow min_{g_t} \quad \frac{\eta}{N} \sum_{n=1}^N (g_t(x_n) -  (y_n - s_n))^2
\end{aligned}
$$

结论,最优的$g_t$就是一个对残差$(y_n - s_n)$进行回归的回归器. 例如用CART树进行回归.

个人思考, 这个思路很绕. 同时不加入正则项,下面的式子就不好优化

$$ min_{g_t} \quad \frac{\eta}{N} \sum_{n=1}^N g_t(x_n) * 2(s_n - y_n) $$

而为什么正则项的大小为$\frac{1}{N} \sum_{n=1}^N \eta g_t(x_n)^2$我认为

* 还是为了后续的推导方便.
* 不需要强回归器

#### 第二个思路

第一个思路中的正则项把我绕了半天. 实际上考虑问题可以有下面简洁的思路.

$$
\begin{aligned}
min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N err(\sum_{\tau=1}^{t-1} \alpha_\tau g_\tau(x_n) + \eta g_t(x_n), y_n) \\
= min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N err(s_n + \eta g_t(x_n), y_n) \\
= min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N (s_n + \eta g_t(x_n) - y_n)^2 \\
= min_{g_t} \quad \frac{1}{N} \sum_{n=1}^N (\eta g_t(x_n) - (y_n - s_n))^2
\end{aligned}
$$

上面的式子很容易推导,结论也很清晰就是对残差$y_n - s_n$的回归, 稍微麻烦的是除了$g_t(x_n)$还有一个$\eta$.

转换一下思路,此时即没有$g_t(x_n)$也没有有$\eta$, 我们实际上应该做的就是构建一个回归器来对残差回归. 也就是设$\eta=1$,也可以想象新的回归器就是$\eta g_t(x_n)$. 当然新的回归器构建好之后在下一步来重新计算新的回归器的权重$\eta$

在AdaBoost中也是同样的思路, 直接用残差构建下一个分类器, 然后再计算新的分类器的权重. 在AdaBoost残差就是下一轮的样本点权重和


### 新加入的回归器的权重

当有了$g_t(x)$后$\eta$的计算就变的简单了,同AdaBoost的优化过程的说明, 用Steepest Descent对$\eta$求偏导并设为0,可以解得$\eta$

$$
\begin{aligned}
min_{\eta} \quad \frac{1}{N} \sum_{n=1}^N (\eta g_t(x_n) - (y_n - s_n))^2 \\
\Rightarrow \frac{1}{N} \sum_{n=1}^N 2(\eta g_t(x_n) - (y_n - s_n)) g_t(x_n) = 0 \\
\Rightarrow \eta = \frac{\sum_{n=1}^N g_t(x_n) (y_n - s_n)}{\sum_{n=1}^N g_t(x_n)^2}
\end{aligned}
$$

### 小结

GBDT 是对AdaBoost DT的一般性扩展,支持任意的代价函数,支持不同分类器或回归器,以回归树为例,计算过程如下

* 初始化$s_0, s_1, ... s_n = 0$ 也就是初始的残差$y_n-s_n = y_n$
* for t=1 ... T
* 用{$x_n, (y_n - s_n)$}构建回归树$g_t$. 例如 bagging采样后训练的裁剪过的CART树
* 计算回归树的权重$\eta_t$,这一步与代价函数相关.
* 更新$s_n = s_n + \eta_t g_t(x_n)$
* end for
* 最后返回$G(x) = \sum_{t=1}^T \eta_t g_t(x)$

### 最后

前两年开始出现的xgboost(extreme gradient boosted tree)十分强悍. 本质思路上同GBDT或者说GBM.

下面贴些别人在网上贴的关于xgboost的说法,来自http://www.jianshu.com/p/2a167d789b73,
()中的为个人理解和猜测

* xgboost在目标函数中加入了正则化项(这个正则项应该与GBDT的有所不同)
* xgboost在迭代优化的时候使用了目标函数的泰勒展开的二阶近似，paper中说能加快优化的过程
* Shrinkage(缩减)，相当于学习速率(xgb中的eta)，xgb在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱各棵树的影响，让后面有更大的学习空间. (思路上没问题,对于GBDT或者GBM来说也应当是弱分类器或弱回归器的加权组合,不要用强分类器来组合)
* 列抽样(column subsampling)，xgb借鉴了rf的做法，支持列抽样，不仅能降低过拟合，还能减少计算. (恩,除了对样本的采样,也对样本的特征进行采样,不过貌似GBDT也应该容易加入这个)
* 对于特征的值有缺失的样本，xgb可以自动学习出它的分裂方向
* xgb支持并行,在特征粒度上并行,xgboost在训练之前，预先对数据进行排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量，这个block结构也使得并行成为了可能，在进行节点分裂时，需要计算每个特征的增益，最终选择增益最大的那个特征去做分裂，那么各个特征的增益计算就可以并行化
* 可并行的近似直方图算法。树节点在进行分裂时，需要计算每个特征的每个分裂点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率会变得很低，所以xgb还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点

我个人用过xgboost几次,感觉很好用,速度也很快. 强烈推荐使用xgboost.
