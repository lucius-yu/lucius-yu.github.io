---
title: 泊松分布,指数分布简介
permalink: /docs/probability/PoissonExponential/
excerpt: poisson and exponential distribution introduction
created: 2018-05-23 22:50:15 +0200
---


## 泊松分布,Poisson分布

泊松分布适合于描述单位时间内随机事件发生的次数的概率分布。如某一服务设施在一定时间内受到的服务请求的次数，电话交换机接到呼叫的次数、汽车站台的候客人数、机器出现的故障数、自然灾害发生的次数、DNA序列的变异数、放射性原子核的衰变数、激光的光子数分布等等

泊松分布的PMF

$$ P(X=k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!} $$

参数说明: $\lambda$ 为单位时间内随机事件的平均发生率, t为有多少个单位时间,k为发生的次数

举例，小明打电动游戏,平均每10分钟杀死一个怪物, 游戏30分钟结束, 问小明在一局游戏中杀死4个怪物的几率是多大?

设单位时间为10分钟,则有
$$ \lambda = 1, t = 3, k = 4 $$

$$ P(X=4) = \frac{3^4 e^{-3}}{4!} = 0.168 $$

泊松分布可以从二项分布导出

将在单位时间内时间发生次数的概率问题转化为二项分布的问题. 将单位时间T分成n份,这里n趋近于无穷大,则可以假设每个T/n的时间小段内时间事件可以为发生或不发生,但是不会有两个事件同时发生. 于是问题就转化为n次实验,每次实验事件发生的概率$p=\frac{\lambda}{n}$,根据二项分布的pmf公式则有泊松分布的pmf如下

$$ \begin{aligned}
P(X=k) = \lim_{n\to\infty} \dbinom{n}{k} (\frac{\lambda}{n})^k (1-\frac{\lambda}{n})^{n-k} \\
= \lim_{n\to\infty} \frac{n!}{(n-k)!k!} (\frac{\lambda}{n})^k (1-\frac{\lambda}{n})^{n-k} \\
= \lim_{n\to\infty} \underbrace{\frac{n!}{(n-k)!k!}}_{F}  (\frac{\lambda}{n})^k \underbrace{(1-\frac{\lambda}{n})^{n}}_{\to e^{-\lambda}} \underbrace{(1-\frac{\lambda}{n})^{-k}}_{\to 1} \\
= \lim_{n\to\infty} \underbrace{\frac{n!}{n^k k!}}_{\to 1}  (\frac{\lambda^k}{k!}) \underbrace{(1-\frac{\lambda}{n})^{n}}_{\to e^{-\lambda}} \underbrace{(1-\frac{\lambda}{n})^{-k}}_{\to 1} \\
= \frac{\lambda^k}{k!}  e^{-\lambda}
\end{aligned}$$

## 指数分布

泊松分布是从二项分布推导而来的，定义为单位时间内时间发生次数的分布。 而指数分布可以轻松地从泊松分布推导而来。

在泊松过程中,设参数为 $\lambda$ (单位时间内发生时间次数服从泊松分布), 单位时间为$t$, 单位时间内时间发生次数$N_t$服从泊松分布

$$P(N_t=k) = \frac{(\lambda t)^k}{k!}  e^{-\lambda t}$$

如果下一个事件要间隔时间t ，就等同于t之内没有任何事件发生.

$$ P(T>t) = e^{-\lambda t} $$

于是得到指数分布的累积分布函数cdf为

$$ P(T<=t) = 1-e^{-\lambda t} $$

求一阶微分可以得到，概率密度函数为

$$ p_T(t) = \frac {d}{dt} (1-e^{-\lambda t}) = \lambda e^{-\lambda t} $$


## 参考

https://zh.wikipedia.org/wiki/%E6%B3%8A%E6%9D%BE%E5%88%86%E4%BD%88
