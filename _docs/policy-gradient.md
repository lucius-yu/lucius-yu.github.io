---
title: 策略梯度
permalink: /docs/rl_learning/policy-gradient/
excerpt: policy gradient in reinforcement learning
created: 2018-08-20 13:40:15 +0200
---

## 说明

Q-learning, Sarsa, DQN都是value-based的强化学习方法, 首先学习到价值函数,然后根据价值函数选择动作.

考虑到Exploration-expolitation(探索与利用)的问题. 一般是采用 $\epsilon$-greedy 的方式选择动作.

输出的动作都是"硬"选择. 本章将讲述另外一类算法, policy-based(基于策略)的强化学习方法.

* 输入为状态,输出为各种动作的概率. 软选择
* 目标是奖励的期望最大化.
* 手段是不断调整策略模型的参数 $\theta$


## 策略梯度方法的优缺点

优点

* 保证局部收敛
* 适用于高位动作空间或者是连续动作空间, 或者是用于难于计算 $arg\max_a Q(s,a)$ 的情况.
* 能学习到随机策略.
  1. 在玩剪刀,石头,布之类的游戏时, 任何确定性策略都不是好策略
  2. 在环境状态不能完全被观测时, 从观测上来说一样结果的实际情况可能是两个或多个不同的状态(状态观测不完全导致不能分辨不同状态), 此时也需要随机策略, 也就是给定状态,执行的动作具有随机性.
* 很多情况下, 比基于学习Q或者V价值函数的方法简单一些. 比如当小球从从空中某个位置落下你需要左右移动接住时，计算小球在某一个位置时采取什么行为的价值是很难得；但是基于策略就简单许多，你只需要朝着小球落地的方向移动修改策略就行。

缺点

* Less sample efficient
* 只能保证局部收敛,而不是全局收敛

## 策略优化目标

我们优化策略的最终目的是什么？尽可能获得更多的奖励。我们设计一个目标函数来衡量策略的好坏，针对不同的问题类*型，这里有三个目标函数可以选择：

* Start value：在能够产生完整Episode的环境下，也就是在个体可以到达终止状态时，我们可以用这样一个值来衡量整个策略的优劣：从某状态s1算起知道终止状态个体获得的累计奖励。这个值称为start value. 这个数值的意思是说：如果个体总是从某个状态s1开始，或者以一定的概率分布从s1开始，那么从该状态开始到Episode结束个体将会得到怎样的最终奖励。这个时候算法真正关心的是：找到一个策略，当把个体放在这个状态s1让它执行当前的策略，能够获得start value的奖励。这样我们的目标就变成最大化这个start value：
$$ J_1(\theta) = V^{\pi_\theta}(s_1) = E_{\pi_\theta} [v_1] $$
* Average Value：对于连续环境条件，不存在一个开始状态，这个时候可以使用 average value。意思是 考虑我们个体在某时刻处在某状态下的概率，也就是个体在该时刻的状态分布，针对每个可能的状态计算从该时刻开始一直持续与环境交互下去能够得到的奖励，按该时刻各状态的概率分布求和：
$$ J_{avV}(\theta) = \sum_s d^{\pi_\theta}(s) V^{\pi_\theta}(s) $$  
注, $d^{\pi_\theta}(s)$ 是在当前策略下马尔科夫链的关于状态的一个静态分布.

* Average reward per time-step：又或者我们可以使用每一个时间步长在各种情况下所能得到的平均奖励，也就是说在一个确定的时间步长里，查看个体出于所有状态的可能性，然后每一种状态下采取所有行为能够得到的即时奖励，所有奖励按概率求和得到：
$$ J_{avR}(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(s,a)R_s^a$$

另外一个策略目标的写法, 这个写法的展开后等于 Average reward per time-step, 不过是强调了调整策略模型参数 $\theta$ 来优化

$$ \max_\theta E_{\pi_\theta} [ \sum_{t=0}^T R(s_t, a_t)] $$

关于这个写法, 要记住  
* $d^{\pi_\theta}(s)$ 在给定策略下,不同状态的分布
* $\pi_\theta(s,a)$  是给定策略下, 在状态s时不同动作a的概率分布,
* $R(s_t, a_t)$ 是t步时给定状态和动作的奖励.

## 策略梯度方法

### 有限差分策略梯度Finite difference Policy Gradient

这是非常常用的数值计算方法，特别是当梯度函数本身很难得到的时候。具体做法是，针对参数θ的每一个分量 $θ_k$，使用如下的公式粗略计算梯度

$$ \frac{\partial J(\theta)}{\partial \theta_k} \approx \frac{J(\theta + \epsilon \mu_k) - J(\theta)}{\epsilon} $$

$u_k$ 是一个单位向量，仅在第k个维度上值为1，其余维度为0. i.e. one-hot向量

有限差分法简单，不要求策略函数可微分，适用于任意策略；但有噪声，且大多数时候不高效. 也能解决问题.

### 蒙特卡罗策略梯度 Monte-Carlo Policy Gradient

#### 似然比方法, likelihood ratio method

* 定义回合(trajectory) $\tau$ 为状态和动作序列 $s_0, a_0, s_1, ..., s_T, a_T$ 并有奖励 $R(\tau) = \sum_{t=0}^T R(s_t, a_t)$
* 定义奖励的期望 $J(\theta) = E[\sum_{t=0}^T R(s_t, a_t); \pi_\theta] = \sum_\tau P(\tau; \theta)R(\tau)$ 可以看出定义就是给定策略参数下,不同回合序列出现的概率乘以序列的奖励. 同时也可以理解如果某种回合(状态和动作序列)的奖励大, 则要调整参数使这种序列出现的概率大.
* 目标, 找到参数 $\theta$ 以最大化下列式子
$$ \max_\theta J(\theta) = \max_\theta \sum_\tau P(\tau; \theta)R(\tau) $$

似然比方法公式推导, 需要梯度上升, 也就是要给出优化目标的梯度函数

$$ J(\theta) = \sum_\tau P(\tau; \theta)R(\tau)  $$

$$ \begin{aligned}
\nabla_\theta J(\theta) = \nabla_\theta \sum_\tau P(\tau; \theta)R(\tau) \\
= \sum_\tau \nabla_\theta P(\tau; \theta)R(\tau) \\
= \sum_\tau \frac{ P(\tau; \theta)}{ P(\tau; \theta)}\nabla_\theta P(\tau; \theta)R(\tau) \\
= \sum_\tau P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} R(\tau) \\
= \sum_\tau P(\tau; \theta) \nabla_\theta logP(\tau; \theta) R(\tau)
\end{aligned}$$

* 在等式中,我们假设可以考虑所有的各种组合的回合(也就是所有可能的状态动作序列组合), 现实中基本是做不到的. 近似的做法是采用策略 $\pi_\theta$ 采样出m个回合. 于是等式被近似为

$$ \nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^m \nabla_\theta logP(\tau^i; \theta) R(\tau^i) $$

* 从上式中可以看出,我们从概率求期望被近似为采样m个回合求平均. 一个好的副作用就是概率P也被从公式中移除.

#### 为什么叫做似然比方法

个人理解

* 似然函数是一种关于统计模型中的参数的函数，表示模型参数中的似然性. 概率用于在已知一些参数的情况下，预测接下来的观测所得到的结果，而似然性则是用于在已知某些观测所得到的结果时，对有关事物的性质的参数进行估计.
* $L(\theta | x_1, x_2, ..., x_n) = P_\theta(x_1, x_2, ..., x_n)$ 给定样本,概率分布参数的似然性为给定参数该特定样本出现的概率.
* 常用的最大似然估计方法就是, 写出概率分布并代入样本数据, 然后取对数, 并分别对不同参数求导后设为0, 取得参数的估计.
* 似然比,给定样本数据, 对不同参数的似然性比值关系
* 这里利用梯度提升,更新前后的参数对应不同的似然性, 似然性的取对数的差值关系对应为原似然性的比值关系.

#### 分解回合为状态和动作序列

$$ \nabla_\theta logP(\tau; \theta) = \nabla_\theta log(\prod_{t=0}^T P(s_{t+1}|s_t, a_t) \pi_\theta (a_t|s_t)) $$
