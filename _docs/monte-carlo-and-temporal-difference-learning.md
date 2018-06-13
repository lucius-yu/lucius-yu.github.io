---
title: 时间差分学习
permalink: /docs/rl_learning/monte-carlo-and-temporal-difference-learning/
excerpt: Monte Carlo learning and Temporal Difference Learning
created: 2018-06-07 03:40:15 +0200
---

## 说明

解决无模型预测和无模型控制的问题. 包含两个部分

* 策略评估 (policy evaluation)
* 策略控制 (policy control)

策略评估用于评估策略的好坏,而策略控制用来改进策略

Model-free learning指我们将解决一个MDP(马尔科夫决策过程)问题,但是我们不知道控制该MDP的模型信息. 而在采用动态规划解决MDP问题时,我们是需要知道模型信息的,例如给定状态和动作,我们知道转移到新的不同状态的概率.

而解决model-free learning的问题，我们讨论

* Monte Carlo learning
* Temporal Difference Learning

### 符号

![notations]({{site.url}}/doc-images/reinforcement-learning/monte-carlo-temporal-difference-learning-01.png)

State-value function 给出在服从策略$\pi$时状态s的真正价值. 从等式中看出是一个递归过程.

## monte-carlo学习的策略评估

既然不知道马尔可夫过程的模型, 直接的想法就是做大量的实验, 从实验中进行统计和参数估计得到模型的估计.
monte carlo学习就是这样的一个直接的解决方法

* 目标是从大量的实验回合中学习到在服从策略 $\pi$ 时的价值方程 $v_{pi}$
$$ S_1, A_1, R_2, ... S_k $$
* 回报(return)为discounted的奖励的累积和
$$ G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1}R_T $$
* 价值方程为回报的期望
* 在MC学习中用实验(样本)均值来代替期望值

### monte-carlo学习的策略评估的基本算法

* 服从策略 $\pi$ 的情况下采样出大量的回合实验
* 在回合中的每个时刻t时,当状态s出现时
  + 计数器加一 $N(s) = N(s) + 1$
  + 累积状态s的回报 $H(s) = H(s) + G_t$
  + 用回报的均值估计价值 $V(s) = H(s)/V(s)$
* 根据大数定律有 $v(s) \rightarrow v_{\pi}(s)$ 当 $N(s) \rightarrow \infty$

增量更新方法

* 增量式的更新价值V
* 对于每个状态s,当其回报为 $G_t$ 有
  + $N(s_t) = N(s_t) + 1$
  + $V(s_t) = V(s_t) + \frac{1}{N(s_t)} (G_t - V(S_t))$


简单的增量公式推导, 为防止符号混淆, 记状态s在t时刻之前有价值V,累积回报G,累积访问计数N.

$$ V = \frac{G}{N}, V(s_t) = \frac{G+G_t(s)}{N+1}, N(s_t) = N + 1 $$

价值增量

$$ V(S_t) - V = \frac{N(G+G_t(s))-G(N+1)}{N(N+1)} = \frac{NG_t(s)-G}{N(N+1)} =  \frac{G_t(s)}{N+1} - \frac{G}{N(N+1)} = \frac{G_t}{N_t} -  \frac{V}{N_t} $$

增量更新

$$ V(s_t) = V + \frac{1}{N_t}({G_t-V}) $$

### first visit 方法

前面说的是每次访问状态s都累积求平均的做法,还有一种是在一个回合中只考虑首次访问状态的累积求和方法。

![first visit MC]({{site.url}}/doc-images/reinforcement-learning/monte-carlo-temporal-difference-learning-02.png)

### 备注

monte-carlo学习的策略评估有一个问题, 那就是学习是要基于完整的回合. 因为更新需要用到回报 $G_t$.
只有当回合结束时才能计算得到


## 时序差分学习

### 简介

* model free 学习
* 价值更新无需等待回合结束,使用bootstrapping
* 使用一个估计(上回合后的t+1的回报估计值)来更新另一个估计.

### MC与TD的对比

![MC and TD]({{site.url}}/doc-images/reinforcement-learning/monte-carlo-temporal-difference-learning-03.png)

从更新公式来看, TD是有偏估计因为V的估计用到了另一个估计量, 而MC是无偏估计,无偏估计量的期望等于真值.

* 样本的均值为无偏估计, 因为样本的均值的期望为真值 $\mu$.
* 而用样本均值 $\overline{X_i}$ 代替真值 $\mu$ 计算样本方差时为有偏估计.
* TD使用了自助法(bootstrapping), 机器学习中bootstrapping指对数据集进行n次有放回的采样而形成的多个样本数据集进行模型训练,例如每个样本集会训练一个模型得到一个估计值，n个样本集会训练n个独立模型,最终的模型聚合了n个独立模型的输出,如随机森林,GDBT等等, 这里广义的指利用估计值进行估计的方法.

关于bootstrapping, 最常用的一种是.632自助法，假设给定的数据集包含d个样本。该数据集有放回地抽样d次，产生d个样本的训练集。这样原数据样本中的某些样本很可能在该样本集中出现多次。没有进入该训练集的样本最终形成检验集（测试集）。 显然每个样本被选中的概率是1/d，因此未被选中的概率就是(1-1/d)，这样一个样本在训练集中没出现的概率就是d次都未被选中的概率，即 $(1-1/d)^d$ . 当d趋于无穷大时，这一概率就将趋近于e-1=0.368，所以留在训练集中的样本大概就占原来数据集的63.2%。

### TD的例子

Driving Home Example

![Driving Home]({{site.url}}/doc-images/reinforcement-learning/monte-carlo-temporal-difference-learning-04.png)

* Predicted Time to Go ($V(S_t)$)是从历史回合中特出的估计值
* 当离开办公室时, 预估总时间为30
* 当到达汽车时,由于下雨, 预估剩余需要时间(类似 $V(S_{t+1})$), 改为35(来自历史经验), 而 $V(S_t)$ 更新为40, 此时TD error为 $G_t - V(S_t) = 43-40 = 3$
* 可以看出TD学习评估策略时,更新不必等到回合结束, 实际上在不断运用以前得到的估计值代替真实的回报
* 第二张图中更新的效果是一样的,但是MC是要等到到家后才进行更新的
* 实际运用中Predicted Time to Go可以是一个监督学习的时序预测,根据天气，不同道路，当前车流等等
* 这个例子中没有列出action供代理选择，譬如 走高速还是不走高速等等.

这里我们利用下一步(时刻t+1)的估计值来更新当前的值, 这是一步时序差分记为TD(0). 而n步时序差分为TD($\lambda$)


收敛性, 对于任何固定的策略 $\pi$ , TD学习得到的v收敛到$v_\pi$

收敛速度比较, 采用MRP过程来对比MC和TD的收敛速度. MRP是没有action的MDP. 一个随机游走的MRP过程,如下图

![Random Walk]({{site.url}}/doc-images/reinforcement-learning/monte-carlo-temporal-difference-learning-05.png)

由中间点C开始,每一步为50%向左或者向右, 移动到最左边或者最右边结束, 如果最右边结束则有奖励1, 其他情况奖励为0.

状态A到E,的真实价值为
$$\frac{1}{6} \frac{2}{6} ... \frac{5}{6}$$

结果如下

![convergence]({{site.url}}/doc-images/reinforcement-learning/monte-carlo-temporal-difference-learning-06.png)

* TD收敛快很多
* 有些TD曲线先下降而后略有上升只是由于最后一步的步长造成的, 一直运行下去的还是会稳定收敛的.


## 参考

http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/MC-TD.pdf
http://www.cnblogs.com/jinxulin/p/3560737.html
