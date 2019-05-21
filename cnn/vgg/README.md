

[Very Deep Convolutional Networks for Large-Scale Image Recognition](<https://arxiv.org/abs/1409.1556>)

![](https://res.cloudinary.com/chenzhen/image/upload/v1558259955/github_image/2019-05-19/vgg.png)

VGG采用的是pre-trained的方式来解决权重初始化的问题，即先训练一小部分网络，当确保这部分网络稳定之后，在此基础上增加网络深度；当训练到D阶段，也就是VGG-16时，发现效果最好，下图为VGG-16具体的网络结构；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558260280/github_image/2019-05-19/vgg-16.png)

**VGG-16:**

* VGG-16包含多个$conv\rightarrow conv \rightarrow max\_pool$这种类似的结构；

* 所有的卷积操作的卷积核的大小都是$3\times 3​$，都是采用same的方式，下采样由最大池化操作来完成；
* 卷积核的数量变化：$64 \rightarrow 128 \rightarrow  256  \rightarrow  512  \rightarrow  512 $

**$3\times 3$的卷积核**

* 两个$3\times 3$的卷积层相当于一个$5\times 5$的卷积，三个$3\times 3$的卷积层相当于一个$7\times 7$的卷积，而且具有更少的参数；
* 多个$3\times 3$的卷积有更多的非线性操作；

**$1\times 1$的卷积核**

* 作用时在不影响输入输出维数的情况下，对输入输出进行线性变换，然后通过ReLU进行非线性处理，增加网络的非线性表达能力；