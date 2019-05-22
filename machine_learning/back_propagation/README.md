---
layout:     post                   
title:      线性回归                
date:       2019-05-12           
categories: 总结
tags:                       
    - 总结
mathjax: true
---

全文参考《机器学习》-周志华中的5.3节-误差逆传播算法；整体思路一致，叙述方式有所不同；

![](https://res.cloudinary.com/chenzhen/image/upload/v1557281094/github_image/2019-05-08/network.jpg)

使用如上图所示的三层网络来讲述反向传播算法；

首先需要明确一些概念，

假设数据集**$X=\{x^1, x^2, \cdots, x^n\}, Y=\{y^i, y^2, \cdots, y^n\}$**，反向传播算法使用数据集中的每一个样本执行前向传播，之后根据网络的输出与真实标签计算误差，利用误差进行反向传播，更新权重；

使用一个样本$(x, y)$，其中$x=(x_1, x_2, \cdots, x_d)$

**输入层：**

&emsp; 有$d$个输入结点，对应着样本$x$的$d$维特征，$x_i$表示输入层的第$i$个结点；

**隐藏层：**

&emsp; 有$q$个结点，$b_h$表示隐藏层的第$h$个结点；

**输出层：**
&emsp; 有$l$个输出结点，$y_j$表示输出层的第$j$个结点；

**权重矩阵：**

&emsp; 两个权重矩阵$V, W$，分别是位于输入层和隐藏层之间的$V\in R^{d\times q}$，其中$v_{ih}$表示连接结点$x_i$与结点$b_h$之间的权重；以及位于隐藏层与输出层之间的$W\in R^{q\times l}$，其中$w_{hj}$表示连接结点$b_h$与结点$y_j$的权重；

**激活函数：**

&emsp; 激活函数使用sigmoid函数；
$$
f(x) = \dfrac{1}{1+e^{-x}}
$$
其导数为：
$$
f'(x) = f(x)(1-f(x))
$$
**其他：**

&emsp; 在隐藏层，结点$b_h$在执行激活函数前为$\alpha_h$，即隐藏层的输入；所以有：
$$
\alpha_h = \sum_{i=1}^{d}v_{ih}x_i
$$
之后经过sigmoid函数：
$$
b_h = sigmoid(\alpha_h)
$$
&emsp; 在输出层，结点$y_j$在执行激活函数前为$\beta_j$，即输出层的输入；所以有：
$$
\beta_j = \sum_{h=1}^{q}w_{hj}b_h
$$
之后经过sigmoid函数：
$$
\hat{y}_j = sigmoid(\beta_j)
$$

### 前向传播

&emsp; 所以，根据上面一系列的定义，前向传播的过程为：由输入层的结点$(x_1, x_2, \cdots, x_i, \cdots, x_d)$，利用权重矩阵$V$计算得到$(\alpha_1, \alpha_2, \cdots, \alpha_h, \cdots, \alpha_q)$，经过激活函数sigmoid得到$(b_1, b_2, \cdots, b_h, \cdots, b_q)$，这就得到了隐藏层的输出；之后，利用权重矩阵$W$计算得到$(\beta_1, \beta_2, \cdots, \beta_j, \cdots, \beta_l)$，经过激活函数sigmoid得到$(\hat{y}_1,\hat{y}_1, \cdots, \hat{y}_j , \cdots, \hat{y}_l )$，也就是最后的输出；

步骤：

**Step 1: **输入层$x \in R^{1\times d}$，计算隐藏层输出$b = sigmoid(x\times V), \quad b\in R^{1\times q}$；

**Step 2: ** 输出层输出$\hat{y} = sigmoid(b \times W), \quad \hat{y}\in R^{1\times l}$；

&emsp; 注意，在前向传播的过程中，记录每一层的输出，为反向传播做准备，因此，需要保存的是$x, b, \hat{y}$； 

前向传播还是比较简单的，下面来看反向传播吧；

### 反向传播

&emsp; 想一下为什么要有反向传播过程呢？其实目的就是为了更新我们网络中的参数，也就是上面我们所说的两个权重矩阵$V, W$，那么如何来更新呢？

> 《机器学习》周志华
>
> BP是一个迭代算法，在迭代的每一轮中采用广义的感知机学习规则对参数进行更新估计，任意参数v的更新估计式为：

$$
v \leftarrow v + \Delta v
$$

> BP算法基于**梯度下降**策略，以目标的负梯度方向对参数进行调整；

我们如何来更新参数呢？也就是如何更新$V, W$这两个权重矩阵；以$W$中的某个参数$w_{hj}$举例，更新它的方式如下：
$$
w_{hj} \leftarrow w_{hj} + \Delta w_{hj}
$$
那么，如何计算$\Delta w_{hj}$的呢？计算如下：
$$
\Delta w_{hj} = -\eta \dfrac{\partial{E}}{\partial{w_{hj}}}
$$
其中，$E_k$表示误差，也就是网络的输出$\hat{y}$与真实标签$y$的均方误差；$\eta$表示学习率；负号则表示沿着负梯度方向更新；
$$
E  = \dfrac{1}{2}\sum_{j=1}^{l}(\hat{y}_j - y_j)^2
$$
也就是说，我们想要对哪一个参数进行更新，则需要计算当前网络输出与真实标签的均方误差对该参数的偏导数，即$\dfrac{\partial{E}}{\partial{w_{hj}}}$，之后再利用学习率进行更新；

在这个三层的网络结构中，有两个权重矩阵$V, W$，我们该如何更新其中的每一个参数呢？

就以权重矩阵$W$中的参数$w_{hj}$来进行下面的解释，

那么根据上面所叙述的，更新$w_{hj}$得方式为：
$$
w_{hj} \leftarrow w_{hj} + \Delta w_{hj}
$$

$$
\Delta w_{hj} = -\eta \dfrac{\partial{E}}{\partial{w_{hj}}}
$$

那么如何来计算$\dfrac{\partial{E}}{\partial{w_{hj}}}$呢？

这里就需要用到**链式法则**了，如果不熟悉的，建议查找再学习一下；
$$
f(x) = g(x)+x
$$

$$
g(x) = h(x) + x^2
$$

$$
h(x) = x^3 + 1
$$

想一下是怎么求$\dfrac{df(x)}{dx}$的；

如果对上文中讲述的网络结构，能够将其完整的呈现的在脑海中的话，对于下面的推导应该不会很困难。

> 再回顾一遍前向传播：
>
> 所以，根据上面一系列的定义，前向传播的过程为：由输入层的结点$(x_1, x_2, \cdots, x_i, \cdots, x_d)$，利用权重矩阵$V$计算得到$(\alpha_1, \alpha_2, \cdots, \alpha_h, \cdots, \alpha_q)$，经过激活函数sigmoid得到$(b_1, b_2, \cdots, b_h, \cdots, b_q)$，这就得到了隐藏层的输出；之后，利用权重矩阵$W$计算得到$(\beta_1, \beta_2, \cdots, \beta_j, \cdots, \beta_l)$，经过激活函数sigmoid得到$(\hat{y}_1,\hat{y}_1, \cdots, \hat{y}_j , \cdots, \hat{y}_l )$，也就是最后的输出；

------

------

那么如何来计算$\dfrac{\partial{E}}{\partial{w_{hj}}}$呢？

我们想一下在网络的均方误差$E$与参数$w_{hj}$之间有哪些过程，也就是说需要想明白参数$w_{hj}$是怎么对误差$E$产生影响的；

$w_hj$是连接隐藏层结点$b_h$与输出层结点$\hat{y}_j$的权重，因此过程是：$b_h \rightarrow \beta_j \rightarrow \hat{y}_j \rightarrow E$

那么根据链式法则就可以有：
$$
\dfrac{\partial E}{\partial w_{hj}} = \dfrac{\partial E}{\partial \hat{y}_j}\dfrac{\partial \hat{y}_j}{\partial \beta_j} \dfrac{\partial \beta_j}{\partial w_{hj}}
$$
分别来求解$\dfrac{\partial E}{\partial \hat{y}_j}$, $\dfrac{\partial \hat{y}_j}{\partial \beta_j}$, $ \dfrac{\partial \beta_j}{\partial w_{hj}}$这三项；

（1）第一项：$\dfrac{\partial E}{\partial \hat{y}_j}$

想一下$E$与$\hat{y}_j$之间有什么关系，即：
$$
E = \dfrac{1}{2}\sum_{j=1}^{l}(\hat{y}_j-y_j)^2 = \dfrac{1}{2}[(\hat{y}_1-y_1)^2+\cdots +(\hat{y}_j-y_j)^2+\cdots+(\hat{y}_l-y_l)^2]
$$
那么，$E_k$对$\hat{y}_j$求偏导：
$$
\dfrac{\partial E}{\partial \hat{y}_j} = -(\hat{y}_j-y_j)
$$
（2）第二项：$\dfrac{\partial \hat{y}_j}{\partial \beta_j}$

再想一下$\hat{y}_j$与$\beta_j$之间有什么关系呢，即
$$
\hat{y}_j = sigmoid(\beta_j)
$$
那么，$\hat{y}_j$对$\beta_j$求偏导，即：
$$
\dfrac{\partial \hat{y}_j}{\partial \beta_j} = \hat{y}_j(1-\hat{y}_j)
$$
（３）第三项：$ \dfrac{\partial \beta_j}{\partial w_{hj}}$

再想一下$\beta_j$与$w_{hj}$之间又有什么关系呢，即：
$$
\beta_j = \sum_{h=1}^{q}w_{hj}b_h= w_{1j}b_1 + \cdots+w_{hj}b_h+\cdots+w_{qj}b_q
$$
所以从上式中能够看清$\beta_j$与$w_{hj}$之间的关系了吧，其实再想一下，$\beta_j$是输出层的第$j$个结点，而$w_{hj}$是连接隐藏层结点$b_h$与结点$\beta_j$的权重；

那么$\beta_j$对$w_{hj}$的偏导数，即：
$$
\dfrac{\partial \beta_j}{\partial w_{hj}} = b_h
$$
上面三个偏导数都求出来了，那么就有：
$$
\dfrac{\partial E}{\partial w_{hj}} = \dfrac{\partial E}{\partial \hat{y}_j}\dfrac{\partial \hat{y}_j}{\partial \beta_j} \dfrac{\partial \beta_j}{\partial w_{hj}}　＝-(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j)b_h
$$
那么更新参数$w_{hj}$
$$
w_{hj} \leftarrow w_{hj}+\Delta w_{hj}
$$

$$
\Delta w_{hj} = -\eta \dfrac{\partial E}{\partial w_{hj}} = -\eta(-(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j)b_h)
$$

即：
$$
w_{hj} = w_{hj}+\Delta w_{hj} =  w_{hj} -\eta(-(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j)b_h)
$$
从上式可以看出，想要对参数$w_{hj}$进行更新，我们需要知道上一次更新后的参数值，输出层的第$j$个结点$\hat{y}_j$，以及隐藏层的第$h$个结点$b_h$；**其实想一下，也就是需要知道参数$w_{hj}$连接的两个结点对应的输出；那么这里就提醒我们一点，在网络前向传播的时候需要记录每一层网络的输出，即经过sigmoid函数之后的结果；**

------

------

现在我们知道如何对权重矩阵$W$中的每一个参数$w_{hj}$进行更新，那么如何对权重矩阵$V$中的参数$v_{ih}$进行更新呢？其中，$v_{ih}$是连接输入层结点$x_i$与隐藏层结点$b_h$之间的权重；

同样是利用网络的输出误差$E_k$对参数$v_{ih}$的偏导，即：
$$
v_{ih} \leftarrow v_{ih} + \Delta v_{ih}
$$

$$
\Delta v_{ih} = -\eta \dfrac{\partial{E}}{\partial{v_{ih}}}
$$

那么如何来计算$\dfrac{\partial{E}}{\partial{v_{ih}}}$呢？想一下是$E$与$v_{ih}$之间有什么关系，过程为：
$$
v_{ih} \rightarrow \alpha_h \rightarrow b_h \rightarrow \beta \rightarrow \hat{y} \rightarrow E
$$
同样是利用链式求导法则，有：
$$
\dfrac{\partial{E}}{\partial{v_{ih}}} = \dfrac{\partial E}{\partial b_h}\dfrac{\partial b_h}{\partial \alpha_h}\dfrac{\partial \alpha_h}{\partial v_{ih}}
$$
同样地，分别来求解$\dfrac{\partial E}{\partial b_h}$,$\dfrac{\partial b_h}{\partial \alpha_h}$,$\dfrac{\partial \alpha_h}{\partial v_{ih}}$这三项；

（1）第一项：$\dfrac{\partial E}{\partial b_h}$

与上述思路相同，想一下$E_k$与$b_h$之间的关系，又可以分解为：
$$
\dfrac{\partial E}{\partial b_h} =\sum_{j=1}^{l}\dfrac{\partial E}{\partial \beta_j} \dfrac{\partial \beta_j}{\partial b_h}
$$
其中，
$$
\dfrac{\partial E}{\partial \beta_j}  = \dfrac{\partial E}{\partial \hat{y}_j}\dfrac{\partial \hat{y}_j}{\partial \beta_j} = -(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j)
$$
另外，$\dfrac{\partial \beta_j}{\partial b_h}$，想一下$\beta_j$与$b_h$的关系：
$$
\dfrac{\partial \beta_j}{\partial b_h} = w_{hj}
$$


所以，就有：
$$
\dfrac{\partial E}{\partial b_h} =\sum_{j=1}^{l}\dfrac{\partial E}{\partial \beta_j} \dfrac{\partial \beta_j}{\partial b_h} = \sum_{j=1}^{l}-(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j) w_{hj}
$$


（2）第二项：$\dfrac{\partial b_h}{\partial \alpha_h}$

同样地，$b_h$与$\alpha_h$之间的关系，有：
$$
b_h = sigmoid(\alpha_h)
$$
那么有：
$$
\dfrac{\partial b_h}{\partial \alpha_h} = b_h (1-b_h)
$$
（3）第三项：$\dfrac{\partial \alpha_h}{\partial v_{ih}}$

同样地，$\alpha_h$与$v_{ih}$之间的关系，有：
$$
\alpha_h = \sum_{i=1}^{d}v_{ih}x_i= v_{1h}x_1 + \cdots+v_{ih}x_i+\cdots+v_{dh}x_d
$$
因此，$\alpha_h$对$v_{ih}$的偏导数为：
$$
\dfrac{\partial \alpha_h}{\partial v_{ih}} = x_i
$$
综合上面三项，有：
$$
\dfrac{\partial{E}}{\partial{v_{ih}}} = \dfrac{\partial E}{\partial b_h}\dfrac{\partial b_h}{\partial \alpha_h}\dfrac{\partial \alpha_h}{\partial v_{ih}} = \sum_{j=1}^{l}-(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j) w_{hj} b_h (1-b_h) x_i
$$

------

------

我们来对比一下$\dfrac{\partial{E}}{\partial{v_{ih}}}$与$\dfrac{\partial E}{\partial w_{hj}}$，两者分别为：
$$
\dfrac{\partial E}{\partial w_{hj}} ＝-(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j)b_h
$$

$$
\dfrac{\partial{E}}{\partial{v_{ih}}} = \sum_{j=1}^{l}-(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j) w_{hj} b_h (1-b_h) x_i
$$

稍微换一种形式，将负号放进去：
$$
\dfrac{\partial E}{\partial w_{hj}} ＝(y_j- \hat{y}_j)\hat{y}_j(1-\hat{y}_j)b_h
$$

$$
\dfrac{\partial{E}}{\partial{v_{ih}}} = \sum_{j=1}^{l}(y_j - \hat{y}_j)\hat{y}_j(1-\hat{y}_j) w_{hj} b_h (1-b_h) x_i
$$

这里我们是对单个参数$w_{hj}, v_{ih}$进行更新，如何对$W, V$整体进行更新呢？

我们再明确一下几个定义：
$x$表示输入层的输出， $x\in R^{1\times d }$；

$b$表示隐藏层的输出，$b\in R^{1\times q }$；

$\hat{y}$表示输出层的输出，$\hat{y}\in R^{1\times l}$；

$sigmoid\_deriv()$表示$sigmoid$的导数，$sigmoid\_deriv(\hat{y}) = \hat{y}(1-\hat{y})$；

将输出层的输出与ground-truth之间的差值记为：$eroor = y-\hat{y}$

可以得到
$$
\dfrac{\partial E}{\partial W} = b' \cdot error \cdot sigmoid\_deriv(\hat{y})
$$

$$
\dfrac{\partial E}{\partial V}= x' \cdot error \cdot sigmoid\_deriv(\hat{y}) \cdot W' \cdot sigmoid\_deriv(b)
$$

在反向传播的过程中，我们记：
$$
D[0] =error \cdot sigmoid\_deriv(\hat{y})
$$

$$
D[1]= error \cdot sigmoid\_deriv(\hat{y}) \cdot W' \cdot sigmoid\_deriv(b)
$$

当将每一个权重矩阵的$D[?]$计算出来，得到一个列表后，再对所有的权重矩阵进行更新；之所以这样做，是为方便代码实现；

### Python实现前向传播与反向传播

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 19-5-7

"""
get started implementing backpropagation.
"""

__author__ = 'Zhen Chen'

# import the necessaty packages
import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # 初始化权重矩阵、层数、学习率
        # 例如：layers=[2, 3, 2]，表示输入层两个结点，隐藏层3个结点，输出层2个结点
        self.W = []
        self.layers = layers
        self.alpha = alpha

		# 随机初始化权重矩阵，如果三层网络，则有两个权重矩阵；
        # 在初始化的时候，对每一层的结点数加1，用于初始化训练偏置的权重；
        # 由于输出层不需要增加结点，因此最后一个权重矩阵需要单独初始化；
        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # 初始化最后一个权重矩阵
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # 输出网络结构
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers)
        )

    def sigmoid(self, x):
        # sigmoid激活函数
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # sigmoid的导数
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display=100):
        # 训练网络
        # 对训练数据添加一维值为1的特征，用于同时训练偏置的权重
        X = np.c_[X, np.ones(X.shape[0])]

        # 迭代的epoch
        for epoch in np.arange(0, epochs):
            # 对数据集中每一个样本执行前向传播、反向传播、更新权重
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # 打印输出
            if epoch == 0 or (epoch + 1) % display == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss
                ))

    def fit_partial(self, x, y):
        # 构造一个列表A，用于保存网络的每一层的输出，即经过激活函数的输出
        A = [np.atleast_2d(x)]

        # ---------- 前向传播 ----------
        # 对网络的每一层进行循环
        for layer in np.arange(0, len(self.W)):
            # 计算当前层的输出
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)

            # 添加到列表A
            A.append(out)

        # ---------- 反向传播 ----------
        # 计算error
        error = A[-1] - y

        # 计算最后一个权重矩阵的D[?]
        D = [error * self.sigmoid_deriv(A[-1])]

        # 计算前面的权重矩阵的D[?]
        for layer in np.arange(len(A)-2, 0, -1):
            # 参见上文推导的公式
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # 列表D是从后往前记录，下面更新权重矩阵的时候，是从输入层到输出层
        #　因此，在这里逆序
        D = D[::-1]

        # 迭代更新权重
        for layer in np.arange(0, len(self.W)):
            # 参考上文公式
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # 预测
        p = np.atleast_2d(X)

        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature
            # matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers int the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking
            # the dot product between the current activation value 'p'
            # and the weight matrix associated wieth the current layer,
            # then passing this value through a nonlinear activation
            # function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the predicted value
        return p

    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # return the loss
        return loss


nn = NeuralNetwork([2, 2, 1])
print(nn.__repr__())
```