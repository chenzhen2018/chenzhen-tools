* Inception v1: [Going deeper with convolutions](<https://arxiv.org/abs/1409.4842>)
* Inception v2: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
* Inception V3: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

## Inception v1

**特点：**

* 自动选择网络结构：Inception module
* 减少模型参数核计算资源：$1\times 1​$的卷积、平均池化（Average Pooling）（去除了全连接层）

**Inception Module**

* 三种不同尺度的卷积：$1\times 1$、$3\times 3$、$5\times 5$，以及$3\times 3$的池化；
* 将四组不同类型但大小相同的特征图并排叠起来，形成新的特征图；
* 降低计算量，同时让信息通过更少的连接传递以达到更加稀疏的特性；
* 不同size的卷积增加了网络对尺度的适应性；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558272411/github_image/2019-05-19/inception_module.png)

**$1\times 1$卷积**

* 减少了参数，允许增加深度；
* 降维，在$3\times 3, 5\times 5$的卷积层前面，压缩或增加或保持通道数；
* 增加网络表达能力；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558272515/github_image/2019-05-19/inceptionv1.png)

## Inception V2

**特点：**

* 借鉴VGG，将Inception Module中的$5\times 5$的卷积替换成两个$3\times 3$的卷积；
* 提出了Bach Normalization层：**目的在于保持网络中每一层的输入属于相同分布**；

### Bach Normalization

**优势：**

* 提升了训练速度，收敛过程大大加快；
* 增加分类效果，类似于Dropout防止过拟合；
* 调参简单，对于参数初始化没有过高要求，可以使用大的学习率；



下面来具体介绍Batch Normalization，BN主要解决的是深度神经网络中的**Internal Covariate Shift**问题；那么什么是**Internal Covariate Shift**呢？

**Internal Covariate Shift**

&emsp; 什么是**Covariate Shift**呢？首先，在统计机器学习中有一个经典的假设：即**训练数据的分布与测试数据的分布是满足相同分布的**；当一个机器学习系统的输入的分布发生变化时，就说明Covariate Shift；

&emsp; 然而在这篇文章中，作者将Covariate Shift的概念进行了扩展，不是应用于整个学习系统，而是这个学习系统的一部分，例如是一个sub-network或layer。怎么用到一个sub-network或者layer中呢？例如：

---

---

有一个network compute：
$$
l =  F_2(F_1(\mu, \Theta_1), \Theta_2)
$$
其中，$l$是损失函数；$F_1, F_2$是任意函数；$\Theta_1, \Theta_2$是需要训练的参数；

假如我们想要训练参数$\Theta_2$，可以看作将$x=F_1(\mu, \Theta_1)$输入到sub-network中：
$$
l = F_2(x, \Theta_2)
$$
执行梯度下降更新参数$\Theta_2​$:
$$
\Theta_2 \leftarrow \Theta_2 - \dfrac{\alpha}{m}\sum_{i=1}^{m}\dfrac{\partial F_2(x_i,\Theta_2)}{\partial \Theta_2}
$$
因此，当训练参数$\Theta_2$的时候，输入的分布（即x的分布）对训练过程影响更大；**如果输入x的分布一直在变化，那么就需要要求参数$\Theta_2$能够学习到该分布的变化规律**，这就多此一举了啊。

**因此，将输入分布固定的话，参数就不需要重新调整适应分布的变化；**



另外，将输入固定还有另外的作用，即**防止发生梯度消失**，怎么应用到的呢？

假如在神经网络中，有一层：
$$
z = g(W\mu+b)
$$
其中，$\mu$表示该层输入，$g$表示sigmoid激活函数；$W, b$表示需要学习的参数；
$$
g(x) = \dfrac{1}{1+\exp(-x)}
$$
我们知道，随着$|x|$的增加，$g'(x)$将会趋近于0，将会发生**梯度消失**；也就是说，对于$x=W\mu+b$中，部分绝对值大的维度将会出现梯度消失的现象；而在训练的过程中，$x$将会受到之前所有层的$W,b$的影响，**不断发生变化的$W,b$将有可能将$x$移动到sigmoid的饱和区域（也就是$g'​$趋近于0）**；

**那么将输入分布固定到sigmoid非饱和区域的话，也就是说将$x=W\mu+b$的分布固定，就不会出现上述梯度消失的情况；**

---

---

&emsp;在这篇文章中，作者将**深度网络中内部结点分布的的改变（the change in the distributions of internal nodes of a deep network）称为Internal Covariate Shift;**

&emsp; **因此，可以看到，固定每一层输入分布的话，有两个作用：一是不需要学习分布变化的规律，二是防止sigmoid时的梯度消失，作者在文中说到：**

> If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

&emsp; 另外，文中也提到其他文章中提出的解决**饱和问题以及梯度消失问题**的解决方案如：ReLU, careful initialization, small learning rates.



**Bach Normalization**

&emsp; 上面说到**固定神经网络中的每一层的输入（即$x=W\mu+b$）**，其实是在激活函数前；

那么该如何固定呢？这就引入了**Batch Normalization**，具体怎么计算呢？

![](https://res.cloudinary.com/chenzhen/image/upload/v1554978079/github_image/2019-04-11/BatchNorm/7.png)

---

---

面试如何回答BN:

&emsp; 首先呢，BN这个算法发表于2015年的ICML，作者提出这个算法的目的是为了解决Internal Covariate Shift问题的；那么什么是Covariate（kouraiv） Shift问提呢？Covariate Shift问题就是表示一个机器学习系统输入的分布是一直在变化的；作者将这个概念进行了扩展，不是用到整个系统的输入，而是一个子网络或者一个layer的输入，这就引入了Internal Covariate Shift问题；那么Covariate Shift问题带来那些弊端呢？这就将导致在训练的过程中，我们的参数能够学习到每一层网络输入分布的变化规律，这就多次一举了。因此，解决该问题的方式就是固定网络中每一层输入的分布；

&emsp; 另外，固定每一层输入的分布还能带来另一个好处；假如我们使用sigmoid作为激活函数的话，能够防止梯度消失；那么如何做到防止梯度消失的呢，我们知道sigmoid函数，当输入x的绝对值比较大的时候，sigmoid的导数将会趋近于0，这就会出现梯度消失，而固定输入的分布到sigmoid的非饱和区域，将会有效地防止发生梯度消失；

&emsp; 最后，BN是如何计算的呢？BN应用于$x=W\mu+b$之后，执行激活函数之前；首先，计算激活值的均值、方差，对激活值进行更新，然后再利用两个参数进行调整更新后的激活值，这就完成了固定输入分布。

BN的优点：稳定的训练，加快收敛，类似于Dropout，简单初始化，可以使用更大的学习率

----

----

### 网络结构

![](https://res.cloudinary.com/chenzhen/image/upload/v1558407394/github_image/2019-05-21/inceptionv2.png)

与Inception v1的不同：

* 将$5\times 5​$的卷积替换成了两个$3\times 3​$的卷积；
* 在Inception Module中，有时使用max poopling，有时使用average pooling；
* $28\times 28$的Inception Module由2增加到3；
* 在任何两个Inception Module中，不再额外使用池化，而是在3c, 4e中使用步长为2的卷积池化；

**Keras实现的过程中遇到的问题：**

* Inception(3c)中，output size为$28\times 28 \times 576$，而使用的步长却是2，这好像有些说不通；因此，在实现的过程中，将Inception(3c)使用步长2，最后output size为$7\times 7 \times 576$；Inception(4e), Inception(5b)也类似；
* 在Inception(4c)：$160+160+160+128=608\ne 576$，**出错**，实现时将128改成96；
* Inception(4b)：与Inception(4c)问题类似；

## Inception V3

**特点：**

* 引入了factorization into small convolutions的思想；
* 优化了Inception Module结构；

**引入Factorization into small convolutions:**

&emsp; 有关Factorization into small convolutions的思想，就是将**一个较大的卷积拆分成两个较小的一维卷积**；如将一个$7\times 7$的卷积拆分成$1\times 7$卷积和$7\times 1$卷积；**一方面节约了大量参数，加速运算并减轻了过拟合；另一方面使用多个非线性的激活函数，扩展了模型的表达能力；**文中提到，**这种非对称卷积结构拆分，其结果比对称地拆为几个相同的小卷积核效果更明显，可以处理更多、更丰富的空间特征，增加特征多样性**；

&emsp; **当feature map的尺度介于12到20之间时，使用这种拆分方式效果更好；**

**优化Inception Module结构：**

&emsp; Inception V3中使用了三种Inception Moudule结构，如下图所示；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558432950/github_image/2019-05-21/inceptionv3_module.png)

**优点：**

* 节省了大量参数，加速运算并减轻了过拟合；
* 扩展了模型的非线性表达能力；
* 处理更多、更丰富的空间特征；



## Inception V4

**Inception V4**

![](https://res.cloudinary.com/chenzhen/image/upload/v1558492883/github_image/2019-05-22/inception_v4.jpg)

**Inception_ResNet_v2**

![](https://res.cloudinary.com/chenzhen/image/upload/v1558492883/github_image/2019-05-22/inception_resnet_v2.jpg)