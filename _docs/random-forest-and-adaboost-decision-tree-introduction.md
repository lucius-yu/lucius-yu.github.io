---
title: 随机森林与Adaptive Boosted决策树简介
permalink: /docs/machine learning/random-forest-and-adaboost-decision-tree/
excerpt: random-forest-and-adaboost-decision-tree introduction
created: 2016-11-06 18:28:26 +0200
---

# 总体概览

* 随机森林
* Adaptive Boosted Decision Tree

这两个模型都是对决策树模型的聚合. 但是出发的思路却是不同的.

## 随机森林

基本算法,简单说就是对决策树做BAGGING.

给定训练数据集$D$, 设一共建$T$棵树.

For t = 1, 2, ... T  

*  从数据集$D$中,用可重复采样方式采样$N$个样本得到数据集$D_t$,这一步是bootstrapping
*  用采样后的数据集建立一个随机化的决策树.这里随机化除了上一步中的随机采样,每一个决策树还可以随机选取数据中的特征(feature)

最后对得到的所有的决策树做uniform的平均

随机森林有几个有趣的性质  

* 可以自带校验self-validation, 这是因为在做bootstrapping采样的时候对于任何一个树都会有一部样本是采样不到的. 这些可以用out-of-bagging的方法来对Validation做出估计
* 可以用permution test对数据集中的不同特征计算出该特征的重要性.
* 对于一个完全长成的决策树,一般来说都是会过拟合的. 所以决策树是需要剪枝的. 而随机森林中的uniform的平均就可以很好的克服过拟合overfitting.

随机森林一般来说就是算比较强的算法, 个人感觉比max margin的SVM可能稍微弱一点. 尽管多个树的平均有近似max margin的效果.

不过随机森林是基于决策树的方法, 所以也继承了决策树的好处.  

* 对于生成的结果可解释性较好
* 对于非数值型的特征比较好处理
* 也能对缺失数据进行处理
* ...

最后的问题是, 能不能改进一下最后的结果不用平均,而是某种加权平均来增强性能呢, 譬如用AdaBoost的聚合方法来增强

## Adaptive Boosted Decision Tree

### 基本算法

给定训练数据集$D$, 设一共建$T$棵树.

For t = 1, 2, ... T  

*  从数据集$D$中,对每个样本计算权重$u^{(t)}$
*  用加权后的样本数据来计算得到一个决策树$DTree(D,u^{(t)})$
*  对获得的决策树计算其投票权重

最后对所有获得的决策树进行加权线性合并.

### 加权的训练错误的两种表达方式

* 一种是用样本的权重 $$ E_{in}^u(h) = \frac{1}{N} \sum_{n=1}^N u_n^t \lvert y_n \neq h(x_n) \rvert$$
* 另一种是用权重采样后得到的数据集表示. 这种表示中,例如一个权重为2的样本,在采样后的数据集中就会有2份 $$ E_{in}^{0/1} = \frac{1}{N} \sum_{x,y \in D_t } \lvert y \neq h(x) \rvert$$

### 样本权重的选择

对于AdaBoost这种聚合模型,关于权重的选择,启发式的思路是去保持单个的分类器多样性的最大化.

这种多样性的最大化是让在第t轮产生的分类器在t+1轮采样出的样本性能非常的不好.这样由t+1轮样本训练出的分类器就与第t轮的分类器非常的不同.

第t轮的分类器

$$ g_t \leftarrow argmin(\sum_{n+1}^N u_n^{(t)} \lvert y_n \neq h(x_n) \rvert $$

第t+1轮的分类器

$$ g_{t+1} \leftarrow argmin(\sum_{n+1}^N u_n^{(t+1)} \lvert y_n \neq h(x_n) \rvert $$

对于$u_n^{(t+1)}$的构建是让$g_t$在$u_n^{(t+1)}$的样本权重下性能很差. 即让下式成立

$$ \frac{\sum_{n=1}^N u_n^{(t+1)} \lvert y_n \neq g_t(x_n) \rvert}{\sum_{n=1}^N u_n^{(t+1)}} = \frac{1}{2} $$

设$g_t$在第t轮的加权错误率为$\epsilon$, 则其正确率为$1-\epsilon$.

$$ \epsilon = \frac{\sum_{n=1}^N u_n^{(t)} \lvert y_n \neq g_t(x_n) \rvert}{\sum_{n=1}^N u_n^{(t)}} $$

则第t+1轮的权重更新规则为:

对于$g_t$分类错误的样本,
$$u_n^{t+1} \leftarrow u_n^{t} (1-\epsilon) $$

对于$g_t$分类正确的样本,
$$u_n^{t+1} \leftarrow u_n^{t} (\epsilon) $$

从上式可以得出,对于$g_t$在新的样本权重下,即$u_n^{t+1}$,其加权错误率为1/2. 这样下一轮训练出的分类器就与$g_t$很不同,并且'着重'点在$g_t$犯错的样本上.

补充:

* 样本的初始权重为均匀分布,每个样本权重都为$1/N$
* 实际使用中是定义缩放因子, 错误样本的权重乘以缩放因子,而正确样本的权重要除以缩放因子.
$$ \diamond_t = \sqrt \frac{1-\epsilon}{\epsilon} $$

缩放因子的意义在于,如果$\epsilon < 1/2$,则'错误样本权重放大'且'正确样本权重减小'

### 线性合并的权重

线性合并的权重计算的思路是对于某个好的分类器其权重大,而差的分类器权重要小. AdaBoost的作者给出的权重计算公式为
$$ \alpha_t = ln(\diamond_t) $$

意义在于  

* 如果$\epsilon=\frac{1}{2}$,则该分类器的权重为0
* 如果$\epsilon=0$,则该分类器的权重为无穷大
* 如果$\epsilon<\frac{1}{2}$,则该分类器的权重为负值

## 总结

随机森林与AdaBoost Decision Tree都是对决策树的聚合模型, 但是两者的思路是有区别的

* 随机森林是用完全长成的树,每个分类器是强分类器,训练错误很小但是是过拟合的.采用平均的方法来克服过拟合,也就是减小variance
* AdaBoost Decision Tree是用弱分类器,每个决策树都是'浅'的树,在生成这些树的时候一般是要限制树的高度. 单个分类器一般都是欠拟合的.

总体感觉AdaBoost Decision Tree的构思更巧妙一些,个人觉得AdaBoost Decision Tree应该性能更好一些

Gradient Boost是AdaBoost的更一般化模型, 支持任意可微分的代价函数.

2014年在kaggle竞赛上出现的xgboost性能很强,多次在kaggle获得很好的结果.
