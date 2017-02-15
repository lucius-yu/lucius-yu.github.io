---
title: 二项分布,负二项分布,泊松分布简介
permalink: /docs/probability/BinomialPascalPoisson/
excerpt:binomial pascal poisson introduction
created: 2017-02-14 22:50:15 +0200
---

## 二项分布

对于经典的抛硬币的实验,给定$\theta$为抛硬币正面向上的概率 关于在n次实验中出现正面向上的次数为k次的二项分布的概率质量函数为

$$p(k \vert n,\theta) = \dbinom{n}{k} \theta^k (1-\theta)^{(n-k)}  = \frac{n!}{k!(n-k)!} \theta^k (1-\theta)^{(n-k)} $$


简单的理解就是n次实验中发生k次正面向上的情况一共有(n选k)种组合. 每中组合发生的概率为$\theta^k (1-\theta)^{(n-k)}$


下图为10次实验正面向上为5次的概率质量函数
![binomial pmf]({{ site.url}}/doc-images/machine-learning/beta-dirichlet-baysian-estimation-01.png)

## 负二项分布, Pascal分布

“负二项分布”与“二项分布”的区别在于：“二项分布”是固定试验总次数N的独立试验中，成功次数k的分布；而“负二项分布”是所有到成功r次时即终止的独立试验中，失败次数k的分布。

由于成功r次时即终止,所以第r次必为成功在前r+k-1次实验中出现r-1次成功,k次失败服从二项分布由此可得

$$p(k \vert n,\theta) = \dbinom{r+k-1}{r-1} \theta^{r-1} (1-\theta)^{k} \theta  = \frac{(r+k-1)!}{k!(r-1)!} \theta^{r} (1-\theta)^{k} $$

当r=1时,负二项分布退化为几何分布

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

## 参考

https://zh.wikipedia.org/wiki/%E6%B3%8A%E6%9D%BE%E5%88%86%E4%BD%88
