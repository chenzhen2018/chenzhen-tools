---
layout:     post                   
title:      MachineLearning_逻辑回归               
date:       2019-05-12           
categories: 总结
tags:                       
    - 总结
mathjax: true
---

> Logistic回归是一种分类算法；

* 假设函数
* 损失函数
* 梯度下降法进行迭代
* 



**Hypothesis Representation:**
$$
h_{\theta}(x) = g(\theta^Tx) = \dfrac{1}{1+e^{-\theta^Tx}}
$$
其中，$\theta \in R^{n+1}, x \in R^{n+1}$；$g(z) = \dfrac{1}{1+e^{-z}}$;

**Cost Function:**
$$
J_{\theta}(x) = -\dfrac{1}{m}[\sum_{i=1}^{m}y^ilog(h_{\theta}(x^i)) + (1-y^i)log(1-h_{\theta}(x^i))]
$$
**Gradient Descent:**
$$
\dfrac{\partial J(\theta)}{\partial \theta_j} = \dfrac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_j^i
$$
**Repeat to converge{**
$$
\theta_j := \theta_j - \alpha \dfrac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_j^i
$$
**}**



**损失函数推导：**

首先，损失函数为：
$$
J(\theta) = \dfrac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2
$$
继续：
$$
J(\theta) = \dfrac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^i),\ y^i)
$$
其中：
$$
Cost(h_\theta(x^i),\ y^i) = \dfrac{1}{2}(h_\theta(x^i)-y^i)^2
$$
公式(7)可以更改为：
$$
Cost(x^i),\ y^i)=\begin{cases}
-log(h_{\theta}(x)) & y=1\\
-log(1 - h_{\theta}(x)) & y=0 \\
\end{cases}
$$
进一步的：
$$
Cost(x^i),\ y^i) = - y^ilog(h_{\theta}(x^i)) - (1-y^i)log(1-h_{\theta}(x^i))
$$
**决策边界：**

- 当$\theta^Tx \geq 0$，即$h_{\theta}(x) \geq 0.5$， 对应结果$y=1$；
- 当$\theta^Tx < 0$，即$h_{\theta}(x) < 0.5$， 对应结果$y=0$；
- **所以$\theta^Tx$是决策边界；**
- **决策边界与假设函数和参数有关，与数据集本身是无关的；**



**过拟合与欠拟合：**

- 过拟合：能够很好的拟合训练集，但是对于新的数据集不具有泛化能力；



**解决过拟合的方法：**

- 减少特征的数量；
- 使用所有特征，但是需要使用正则化；



**正则化：**

- 得到一个更简单一些的假设函数；
- 减少过拟合；

### 正则化后的Logistic回归：

**Hypothesis Representation:**
$$
h_{\theta}(x) = g(\theta^Tx) = \dfrac{1}{1+e^{-\theta^Tx}}
$$
其中，$\theta \in R^{n+1}, x \in R^{n+1}$；$g(z) = \dfrac{1}{1+e^{-z}}$;

**Cost Function:**
$$
J_{\theta}(x) = -\dfrac{1}{m}[\sum_{i=1}^{m}y^ilog(h_{\theta}(x^i)) + (1-y^i)log(1-h_{\theta}(x^i))] + \dfrac{\lambda}{2m}\sum_{i=1}^{m}\theta_j^2
$$
**注意：正则化减小只是$\theta_j， j=1,\cdots,m$；**

**Gradient Descent:**
$$
\dfrac{\partial J(\theta)}{\partial \theta_j} = \dfrac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_j^i + \dfrac{\lambda}{m}\theta_j
$$
**Repeat to converge{**
$$
\theta_0 := \theta_0 - \alpha \dfrac{1}{m}\sum_{i=1}^{m}[h_{\theta}(x^i)-y^i] \qquad j=0
$$

$$
\theta_j := \theta_j - \alpha \dfrac{1}{m}\sum_{i=1}^{m}[(h_{\theta}(x^i)-y^i)x_j^i + \lambda \theta_j] \qquad j=1, \cdots ,m
$$

$$
\theta_j := \theta_j(1-\alpha \dfrac{\lambda}{m}) - \alpha \dfrac{1}{m}[\sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_j^i] \qquad j=1, \cdots ,m
$$

**}**