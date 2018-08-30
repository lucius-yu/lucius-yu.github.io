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
* 很多情况下, 比基于学习Q或者V价值函数的方法简单一些.

缺点

* Less sample efficient
* 只能保证局部收敛,而不是全局收敛
