> 线性回归，就是能够利用一条直线和一个平面精确地描述数据之间的关系；当有新的数据时，可以预测出一个简单的值；

* 假设函数
* 定义损失函数（平方损失、正则化）
* 梯度下降法、正规方程法
* 梯度下降法与正规方程法的优缺点

### 单变量线性回归

Hypothesis:
$$
h_\theta(x) = \theta_0 + \theta_1x
$$
Cost Function(平方误差损失):
$$
J(\theta_0, \theta_1) = \dfrac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2
$$

$$
\mathop{minise}\limits_{\theta_0, \theta_1}\quad J(\theta_0, \theta_1)  
$$

**梯度下降法：最小化代价函数**(Gradient Descent Algorithm)

**step 1:**

&emsp; 初始化$\theta_0, \theta_1$;

**step 2:**

&emsp; 同时更新$\theta_0, \theta_1$，减小$J(\theta_0, \theta_1)$，直到最小，一直迭代：
$$
\theta_j = \theta_j-\alpha \dfrac{\partial}{\partial\theta_j}J(\theta_0, \theta_1) \qquad for\ j=0\ and\ j=1
$$

- 学习率太小：需要很多步才可以达到局部最小；
- 学习率太大：可能会导致无法收敛或者发散；
- 使用固定学习率的时候，在接近局部最优点时，梯度下降法会自动采取更小的更新幅度；这是因为**导数项的值会变小**；

**将损失函数代入到更新策略中，得到：**
$$
\theta_j = \theta_j-\alpha \dfrac{\partial}{\partial\theta_j} \dfrac{1}{2m} \sum_{i=1}^{m}(\theta_0 +\theta_1x^i-y^i) \qquad for\ j=0\ and\ j=1
$$

$$
\theta_0 := \theta_0 - \alpha \dfrac{1}{m}\sum_{i=1}^{m}[h_{\theta}(x^i)-y^i]
$$

$$
\theta_1 := \theta_1 - \alpha \dfrac{1}{m}\sum_{i=1}^{m}[h_{\theta}(x^i)-y^i]x^i
$$

**线性回归的代价函数是凸函数，有全局最优，没有局部最优；**



### 多变量线性回归

Hypothesis:
$$
h_{\theta}(x) = \theta_0x_0 + \theta_1x_1 + \cdots + \theta_nx_n = \theta^Tx
$$

$$
x \in R^{n+1}, \quad \theta \in R^{n+1}
$$

Cost Function:
$$
J(\theta_0, \theta_1, \cdots, \theta_n) = \dfrac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2
$$
**梯度下降法：最小化代价函数**(Gradient Descent Algorithm)

**step 1:**

&emsp; 初始化$\theta_0, \theta_1, \cdots, \theta_n $;

**step 2:**

&emsp; 同时更新$\theta_0, \theta_1, \cdots, \theta_n$，减小$J(\theta_0, \theta_1, \cdots, \theta_n)$，直到最小，一直迭代：
$$
\theta_j = \theta_j-\alpha \dfrac{\partial}{\partial\theta_j}J(\theta_0, \theta_1, \cdots, \theta_n) \qquad for\ j=0,1, \cdots, n
$$
**因此，将损失加入参数更新策略中，得到：**
$$
\theta_j := \theta_j - \alpha \dfrac{1}{m}\sum_{i=1}^{m}[h_{\theta}(x^i)-y^i]x_j^i \qquad for\ j=0,1, \cdots, n
$$

### 在使用梯度下降的时候，一些技巧

- **特征缩放，均值归一化**：也就是将特征$x^i$的值，都缩放到固定范围；
- **调整学习率**：如果梯度算法正常工作的话，每迭代一次，代价函数都会降低；可以通过绘制**迭代次数与损失函数之间的变化曲线**，来判断是否收敛；

### 多项式回归

例如：
$$
h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2^2
$$

- 可以将多项式回归转成线性回归；
- 使用多项式回归，在执行梯度下降法前，有必要使用特征缩放；



### 正规方程

**可以通过正规方程求解参数$\theta$；**
$$
J(\theta) = X\theta - y
$$
通过$minise \  J(\theta)$，可以得到：
$$
\theta = (x^Tx)^{-1}x^Ty
$$
其中，$X \in R^{m \times (n+1)}, x^i \in R^{n+1}, \theta \in R^{n+1}, y \in R^m$；

**正规方程不需要特征缩放；**



|          | 梯度下降法                                           | 正规方程法                                                |
| -------- | ---------------------------------------------------- | --------------------------------------------------------- |
| **缺点** | 需要给定学习率，增加额外的工作；需要进行很多次迭代； | 不需要指定学习率；不需要多次迭代；                        |
| **优点** | 当n很大时，效果也好；                                | 当n很大时，很耗时，因为正规方程法中需要计算**矩阵的逆**； |



### 正则化后的线性回归

**梯度下降法：**

**Hypothesis Function:**
$$
h_{\theta}(x) = \theta^Tx
$$
**Cost Function:**
$$
J(\theta) = \dfrac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^i)-y^i)^2 + \dfrac{\lambda}{2m} \sum_{i=1}^{m}\theta_j^2
$$
**注意：正则化减小只是$\theta_j， j=1,\cdots,m$；**

**Gradient:**
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

**正规方程法：**

![](https://res.cloudinary.com/chenzhen/image/upload/v1552455721/github_image/2019-03-13/%E6%8D%95%E8%8E%B7.png)

矩阵大小是$(n\times 1)\times  (n \times 1)$；

