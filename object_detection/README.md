## Object Detection(RCNN, SPPNet, Fast RCNN, Faster RCNN, YOLO v1)

RCNN -> SPPNet -> Fast-RCNN -> Faster-RCNN -> FPN

YOLO v1-v3

### Reference

* **RCNN:** [Rich feature hierarchies for accurate object detection and semantic segmentation](<https://arxiv.org/abs/1311.2524>)
* **SPPNet:** [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](<https://arxiv.org/abs/1406.4729>)
* **Fast R-CNN:** [Fast R-CNN](<https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf>)
* **Faster R-CNN:** [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](<https://arxiv.org/abs/1506.01497>)（[official code: Caffe](<https://github.com/rbgirshick/py-faster-rcnn>), [official code: MATLAB](<https://github.com/ShaoqingRen/faster_rcnn>)）
* **FPN:** [Feature Pyramid Networks for Object Detection](<https://arxiv.org/abs/1612.03144>)（[official code: Caffe](<https://github.com/unsky/FPN>)）
* **YOLO v1:** [You Only Look Once: Unified, Real-Time Object Detection](<https://arxiv.org/abs/1506.02640>)（[official code: Darknet](<https://pjreddie.com/darknet/yolov1/>)）
* **YOLO v2:** [YOLO9000: Better, Faster, Stronger](<https://arxiv.org/abs/1612.08242>)（[official code: Darknet](<https://pjreddie.com/darknet/yolo/>)）
* **YOLO v3:** [YOLOv3: An Incremental Improvement](<https://arxiv.org/abs/1804.02767>)（[official code: Darknet](<https://pjreddie.com/darknet/yolo/>)）
* 技术博客：[基于深度学习的目标检测技术演进：R-CNN、Fast R-CNN、Faster R-CNN](https://www.cnblogs.com/skyfsm/p/6806246.html)
* 技术博客：[论文笔记：目标检测算法（R-CNN，Fast R-CNN，Faster R-CNN，FPN，YOLOv1-v3）](https://www.cnblogs.com/liaohuiqiang/p/9740382.html)
* 技术博客：[FPN详解](<https://blog.csdn.net/WZZ18191171661/article/details/79494534>)
* 技术博客：[从YOLOv1到YOLOv3，目标检测的进化之路](<https://blog.csdn.net/guleileo/article/details/80581858>)
* 技术博客：[yolo系列之yolo v3【深度解析】](<https://blog.csdn.net/leviopku/article/details/82660381>)

---

---

### RCNN

![](https://res.cloudinary.com/chenzhen/image/upload/v1558494917/github_image/2019-05-22/RCNN.png)

**步骤：**

1. 输入图像；

2. 使用生成region proposals的方法，如selective search，提取约2k个候选区域；

3. warped region，将提取的候选区域固定大小

4. 将候选区域输入到已经训练好的CNN网络模型中提取特征向量；

5. 使用提取的特征向量输入到已经训练好的SVM分类器中，判定输入图像是否属于该类；

6. 为准确定位，训练一个bounding box regression用于精确定位；

**缺点：**

1. 多阶段：训练CNN、训练SVM、训练bounding box regression

2. 空间和时间代价高

3. 预测阶段慢

---

---

### SPP Net（Spatial Pyramid Pooling）

有两个特点：

1. **CNN可以接收不同尺度的输入；**

2. **对原图只提取一次特征；**

**特点1：CNN可以接收不同尺度的输入：**

&emsp; 在RCNN中，需要对提取的候选框执行warped region固定大小，为什么要这样做呢？这是因为CNN模型需要接收固定大小的输入，对于CNN模型中的卷积层来说，其参数于输入大小无关；而对于全连接层来说，其参数与输入大小是有关的，一旦全连接层固定，其输入的大小也必须是固定的；**因此，传统的CNN模型中，其输入大小是固定的；**

&emsp; SPPNet就是为了让CNN模型能够接收不同尺度大小的输入，最后有着固定尺度的输出；具体做法是：**在卷积层和全连接层之间，增加了金字塔层；**

&emsp; 具体做法如下图所示；也就是说，对任意一个$w\times h\times C$的feature map：

1. 首先，划分成$4\times 4$的grid cell，提取每个grid cell的最大值，得到$16\times C$的特征；

2. 其次，划分成$2\times 2$的grid cell，提取每个grid cell的最大值，得到$4\times C$的特征；

3. 最后，划分成$4\times 4$的grid cell，提取每个grid cell的最大值，得到$1\times C$的特征；

4. 因此，最后输出为$21\times C$的特征；

因此，金字塔层就实现了任意尺度特征的输入，可以得到固定尺度的输出；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558497782/github_image/2019-05-22/SPPNet-PyramidPooling.jpg)

**特点2：对原图只提取一次特征：**

&emsp; 在RCNN中，作者首先利用region proposal提取输入图片的候选框，然后将候选框固定大小后，将其送入与训练好的CNN模型提取特征，之后根据提取的特征进行分类；而在SPPNet中，作者同样先提取候选框，然后将整张图片输入到CNN中得到feature map，然后根据候选框的位置找到feature map相对应的patch，将patch作为每个候选框的卷积特征输入到SPP layer（上面提到的金字塔层）中，之后执行后面的全连接层，提取最终的特征；

相比于RCNN，SPPNet只做了一次CNN计算，但仍然存在一些RCNN中的缺点；

**缺点：**

1. 多阶段：训练CNN、训练SVM

2. 空间、时间消耗大

---

---

### Fast RCNN

&emsp; 为了解决RCNN上的**多训练、空间时间消耗大**的缺点，并借助SPPNet的优点，作者提出了Fast RCNN，进一步提升了速度和准确率；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558502685/github_image/2019-05-22/Fast_RCNN.png)

从上图可以看到：

1. 首先，将整张图片输入到CNN中，得到Conv feature map；

2. 其次，根据region proposal提取的候选框，得到候选框的ROC projection；

3. 其次，将ROI projection送入到ROI pooling layer以及两个全链接层得到固定尺度的输出ROI feature vector；

4. 之后分成两个阶段，分别跟上一个全连接层，分类分类和回归；

因此，Fast RCNN与RCNN相比：

**相同点：**

1. 都是使用region proposal的selective search的方式提取候选框；

**不同点：**

1. RCNN对每一个候选框计算CNN得到特征向量，并利用SVM分类，并借助bounding box regression进行精确定位；

2. Fast RCNN只计算一次CNN，并借助ROI pooling layer+classification、regression，进行分类定位；

---

---

### Faster RCNN

&emsp; 在Fast RCNN中，一个最大的瓶颈就是第一步候选框的提取，即selective search，这一步骤非常耗时，有没有更高效的方法来提取这些候选框呢？而Faster RCNN主要就是做了这个工作；**提出一个网络（Region Proposal Network，RPN）来找到这些候选框；**

&emsp; 如下图所示，在卷积层后面加上一个Region Proposal Network可以得到的ROI；之后将这些ROI进行ROI池化，之后就与Fast RCNN一致了，加上全连接层，softmax分类器和回归器；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558504290/github_image/2019-05-22/Faster_RCNN.png)

**RPN：**

具体RPN网络怎么做到提取这些ROI的呢？见下图；可以看到，在最后一个卷积层上以$n\times n$的窗口进行滑动，每个窗口被映射成一个固定维度的特征，然后飞车呢个两路接两个$1\times 1$的卷积得到分类层和回归层；其中，每一个滑动窗口都提取k个anchor；因此，分类层中由2k个scores，表示是或不是这一类；回归层中由4k个coordinates，对应着每一个anchor中目标对应的坐标位置；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558504523/github_image/2019-05-22/Faster_RCNN_RPN.png)

**RPN训练：**

* 正样本：与ground truth的IOU最大的anchor视为正样本，与ground truth的IOU大于0.7的anchor视为正样本；
* 负样本：与ground truth的IOU低于0.3的视为负样本；
* 其他的忽略，超过原图边界的anchor也忽略；
* 整个网络存在四个损失函数，如下图：

![](https://res.cloudinary.com/chenzhen/image/upload/v1558505328/github_image/2019-05-22/Faster_RCNN_loss.png)

### RCNN -> SPP Net -> Fast RCNN -> Faster RCNN：

| 项目       | 传统方法   | R-CNN            | Fast R-CNN       | Faster R-CNN |
| ---------- | ---------- | ---------------- | ---------------- | ------------ |
| 提取候选框 | 穷举搜索   | Selective Search | Selective Search | RPN          |
| 提取特征   | 手动提特征 | CNN              | CNN+ROI          | CNN+ROI      |
| 特征分类   | SVM        | SVM              | CNN+ROI          | CNN+ROI      |

**RCNN:**

1. 在图像总确定约1000~2000个候选框(selective search)；
2. 每个候选框内图像块缩放至相同大小，并输入到CNN内进行特征提取；
3. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类；
4. 对于属于某一特征的候选框，用回归器进一步调整其位置；

**Fast RCNN:**

1. 在图像中确定1000~2000个候选框（）selective search）；
2. 对整张图片输进CNN，得到feature map；
3. 找到每个候选框在feature map上的映射patch，将此patch作为每个候选框的卷积特征输入到SPP layer 和之后的层
4. 对候选框中提取出的特征，使用分类器判别是否属于某一个特定类；
5. 对于属于某一特征的候选框，用回归器进一步调整其位置；

**Faster RCNN：**

1. 对整张图片输进CNN，得到feature map；
2. 卷积特征输入到RPN，得到候选框的特征信息；
3. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类；
4. 对于属于某一特征的候选框，用回归器进一步调整其位置；

---

---

### FPN

---

高斯金字塔：

&emsp;通过高斯平滑和亚采样获得一些采样图像，也就是说第K层高斯金字塔通过平滑、亚采样操作就可以获得K+1层高斯图像，高斯金字塔包含了一系列低通滤波器，其截至频率从上一层到下一层是以因子2逐渐增加，所以高斯金字塔可以跨越很大的频率范围。

&emsp; 总之，输入一张图片，可以获得多个不同尺度的图像，就可以得到类似金字塔形象的一个图像金字塔；通过这个操作，可以为2维图像增加一个尺度维度，这就就可以从中获得更多的有用信息；

&emsp; 高斯金字塔可以在一定程度上提高算法的性能；

**特征金字塔：**

&emsp; **在图像中，存在不同尺寸的目标，而不同的目标具有不同的特征，利用浅层的特征可以将简单的目标区分开，利用深层的特征可以将复杂的目标区分开；这样就可以构造一个不同层次的特征金字塔来将不同的目标进行区分；**

---

**FPN(Feature Pyramid Network)**

&emsp; 识别不同大小的物体是计算机视觉中的一个基本挑战，常用的解决方案是构造多尺度金字塔；那么作者是如何提出FPN的呢？如下图所示，表示四种特征金字塔；

**图a中**，先对原始图像构造图像金字塔，然后在图像金字塔的每一层中提取出不同尺度的特征，然后进行相应的预测bbx的位置信息。这种方法的**缺点**是计算量大，需要大量的内存；而**优点**是可以获得较好的精度；

**图b中**，这是一种改进的思路，学者们发现可以利用卷积网络的特性，通过对原始图像进行一系列的卷积池化操作，可以得到不同尺度大小的feature map；就相当于在图像的特征空间中构造特征金字塔；实验表明，浅层的特征更关注于图像的细节信息，高层的特征更关注于图像的语义信息，而高层的语义信息能够保准我们准确地的检测出目标；因此，利用最后一个卷积层上的feature map（也就是特征金字塔的最顶层特征）进行预测；这种方法被大多数深度网络采用，如VGG，ResNet，Inception等，都是利用最后一层特征进行分类；这种形式的金字塔的**优点**是速度快、需要内存少。而**缺点**是仅仅关注于最后一层特征（即特征金字塔最顶层特征），忽略了特征空间中其他层的特征（也就是特征金字塔的其他层特征）；

**图c**，根据图b中的缺点，就引出了图c；像SSD就采用了这种多尺度特征融合的方式，没有采用上采样过程，即从网络不同层中提取不同尺度的特征做预测，这种方式不会增加额外的计算量；而本文作者认为SSD算法没有用到足够低层的特征；

**图d**，作者提出的特征金字塔如图d所示，顶层特征通过上采样和底层特征做融合，而且每层都是独立预测的，下面将具体介绍；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558526784/github_image/2019-05-22/FPN.png)

&emsp;从上图d中，FPN算法具体是怎么计算的呢？首先，利用利用一个卷积网络对输入图像进行一系列的卷积池化计算，然后提取不同卷积层次的的feature map组成特征金字塔，如图d(左)所示；我们要根据这个特征金字塔来构造一个新的特征金字塔，如图d(右)所示；

&emsp;怎么构造呢？首先，新的特征金字塔(图d(右))顶层特征与之前的特征金字塔的顶层特征一致的；然后，计算新的特征金字塔的次顶层特征，利用新的特征金字塔的顶层特征进行上采样操作得到尺度与**之前特征金字塔次顶层特征**尺度一致，那么重点来了，就将上采样得到的特征与之前特征金字塔的次顶层特征进行融合，这就得到了新的特征金字塔的次顶层特征；（哈哈，大写的尴尬。。。）为构造新的金字塔的其他层，继续上述操作即可；

&emsp; 那么具体的，如何来计算呢？如下图所示；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558530029/github_image/2019-05-22/FPN_2.png)

---

---

### YOLO v1

&emsp; 与R-CNN系列算法（产生候选区，分类，回归修正）不同，YOLO系列算法用一个单独的卷积网络输入图像，输出bounding box以及分类概率；

**优点：**

1. 速度更快；
2. 站在全局的角度进行预测，网络的输入是整张图片，更小的背景误差；
3. 能够学习到泛化能力更强的特征；

**缺点：**

1. 加了很强的空间限制，导致系统能预测的目标数量有限，对靠的近的物体和很小的物体的检测效果不好；
2. 难以泛化到新的不常见的长宽比物体；
3. 损失函数同等对待小的bbx和大的bbx；
4. 准确率无法达到state-of-art水平；

**思想：**

1. 将输入图像划分成$S\times S​$个grid cell，待检测目标（Object）的中心落在哪个grid cell中，就表明这个grid cell**负责**预测这个Object；
2. 每一个grid cell负责预测B个bounding box和C个类别的条件概率；**首先**，在每一个grid cell中，每一个bounding box包含5个值，分别是$x, y, w, h, confidence\ score$；$x, y$相对于grid cell的offset归一化到$[0,1]$，$w,h$相对于整张图归一化到​$[0,1]$，表示bounding box的位置信息；$confidence \ score = Pr(Object) * IOU_{pred}^{truth}$，表示该bounding box的置信度；其中，该bbx中存在Object，则$Pr(Object)=1$，否则为0；$IOU_{pred}^{truth}$表示bbx与ground truth的IOU；**其次**，C个类别的条件概率$Pr(Class_i|Object)$表示每一个grid cell中存在Object的条件下，各个类别的概率；
3. 文章中，取$S = 7, B = 2, C = 20$，因此，网络输出是：$S\times S \times (5\times B + C)=7\times 7\times 30$；
4. 在**测试**的时候，计算如下：表示每一个bbx的class-specific confidence score，可以设置阈值，过滤掉得分较低的bbx；

$$
Pr(Class_i|Object) * Pr(Object) * IOU_{pred}^{truth} = Pr(Class_i)*IOU_{pred}^{truth}
$$

5. 最后，一个物体可能由多个bbx来预测，采用NMS得到最终的检测结果；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558508224/github_image/2019-05-22/YOLOv1.png)

**网络结构：**

&emsp; YOLO v1的网络结构如下图所示；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558509595/github_image/2019-05-22/YOLOv1_network.png)

**训练：**

1. **预训练：**使用上图中的前20层卷积后面加上一个average pooling和全连接层的网络结构，在ImageNet1000上进行预训练；（ImagenNet 2012 validation set: 88% top-5）
2. **detection：**在上述预训练得到的卷积层的基础上，加上四个卷积层和两个池化层（参数随机初始化）进行训练；由于detection更加细粒度的信息，因此网络的输入由$224\times 224$调整到$448\times 448$；
3. 最后一层输出class probabilities & bounding box coordinates，维度为$7\times 7\times 30$；
4. 另外，最后一层使用linear activation function，其他层使用leaky rectified linear activation function（leaky ReLU）；
5. 参数设置：epochs=135, batch_size=64, momentum=0.9, decay=0.0005；学习率设置：在第一个epoch中，学习率由0.0001上升到0.001，继续以0.001训练到75 epochs，然后以0.0001训练30个epochs，最后以0.00001训练30个epochs；

**损失函数：**

&emsp; 整体使用sum-squared error(SSE)作为损失函数，如下图；

![](https://res.cloudinary.com/chenzhen/image/upload/v1550838241/github_image/2019-02-22/2019-02-22-4.jpg)

这里存在几点**问题：**：

1. 简单的使用SSE作为损失函数，8维的localization error与20维的classification error平等地对待，这显得很不合理；
2. 在输入图像中，存在大量的不包含Object的grid cell，对应的bbx的confidence将会被push到0，这很明显是overpowering的，会影响到那些包含Object的grid cell中bbx的confidence的梯度更新；**这将导致模型变的很不稳定；**
3. 简单的SSE损失平等地对待大的bbx与小的bbx，这也显得不合理；损失函数应该能够反应出**在大的bbx中的小偏差比小的bbx的小偏差要小；**

> Our error metric should reflect that small deviations in large boxes matter less than in small boxes.

**改进：**

1. 问题1、2：对于那些**不包含Object的bbx的confidence的损失**给予一个较小的权重，即$\lambda_{noobj}=0.5$；对于那些**包含Object的bbx的坐标损失**给予一个较大的权重，即$\lambda_{coord}=5$；

2. 问题3：在损失函数中，对于bbx的$w, h​$，使用它们的平均根来代替它们本身；如下图，符合原文描述；

> Our error metric should reflect that small deviations in large boxes matter less than in small boxes.

![](https://res.cloudinary.com/chenzhen/image/upload/v1550839203/github_image/2019-02-22/2019-02-22-5.png)

### YOLO v2

&emsp; 这篇文章中提到了两个模型YOLO v2和YOLO 9000；

**YOLO v2：**

&emsp; YOLO v2是在YOLO v1的基础上增加了一系列的设定，如下图所示；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558514171/github_image/2019-05-22/YOLOv2.png)

1. **Batch Normalization:** 使用Batch Normalization层，移除dropout，用于模型正则化；
2. **hi-res classifier:** 表示在正式训练检测网络（$448 \times 448$）之前先使用$448\times 448$输入图像的分类网络进行预训练（在这之前已经在ImageNet上进行224输入的图像的预训练）
3. **convolutional with Anchor Boxes:** 表示借鉴Faster RCNN的做法，去掉回归层之前的全连接层和池化层，在最后一层卷积上进行anchor boxes的预测。这种做法，降低了mAP，提高了Recall；在YOLO v1中，类别概率是由grid cell负责的，而在YOLO v2中，类别概率由anchor box负责，即最后输出的tensor的带下是$S\times S \times B \times (5 + C)​$；同时，调整网络输入为$416\times 416​$；
4. **new network:**  在YOLO v2中使用了新的网络模型，提出了Darknet 19；包含19个卷积，5个最大池化；如下图所示；
5. **dimension priors:** 利用k-meas的方式来设计anchor box；
6. **location prediction:** 不使用Faster RCNN中的offset，而是沿用YOLO v1直接预测相对grid cell的归一化坐标；使用logistic activation进行限制；如图所示，公式如下：

![](https://res.cloudinary.com/chenzhen/image/upload/v1558569993/github_image/2019-05-23/YOLOv2_bbx.png)
$$
\begin{equation}
b_x = \sigma(t_x) + c_x\\
b_y = \sigma(t_y) + c_y \\
b_w = p_we^{t_w}\\
b_h = p_he^{t_h}\\
Pr(Object) * IOU(b, object) = \sigma(t_o)
\end{equation}
$$
从上述图中和公式中，其中$t_x, t_y,t_w, t_h,t_o$表示网络输出的五个量，表示bbx的坐标信息以及置信度，不过仍需要做进一步的处理，才能真正得到bbx的坐标位置和置信度；$\sigma$表示logistic activation用于限制网络的预测到$[0, 1]​$；

为了得到bbx的中心坐标$(b_x, b_y)$，计算方式如下：
$$
\begin{equation}
b_x = \sigma(t_x) + c_x\\
b_y = \sigma(t_y) + c_y \\
\end{equation}
$$
其中，$c_x, c_y$表示当前grid cell的左上角坐标相对于图像左上角的偏移量，而我们$\sigma$则是对网络输出的$(t_x, t_y)$进行限制；

为了得到bbx的宽和高$(b_w, b_h)$，计算方式如下：
$$
\begin{equation}
b_w = p_we^{t_w}\\
b_h = p_he^{t_h}\\
\end{equation}
$$
其中，$(t_w, t_h)$表示网络输出的量，$p_w, p_h$表示kmeans聚类之后的prior模板框的宽和高；

为了得到置信度$Pr(Object)*IOU(b, object)$，计算如下：
$$
\begin{equation}
Pr(Object) * IOU(b, object) = \sigma(t_o)
\end{equation}
$$
其中，$t_o$不表示网络输出，$\sigma$进行限制；

7. **passthrough:** 最后输出的特征图的大小是$13\times 13$，YOLO v2加上了一个Passthrough Layer来取得之前某个$26\times 26​$的feature map；Passthrough Layer 能够将高分辨特征和低分辨特征连接起来；
8. **multi-scale:** 在迭代的过程中，网络会随机选择一个新的输入尺寸进行训练；
9. **hi-res detector:** ？

![](https://res.cloudinary.com/chenzhen/image/upload/v1558515049/github_image/2019-05-22/YOLOv2_Darknet.png)

**YOLO 9000：**

&emsp; YOLO 9000以YOLO v2为主网络，通过联合优化分类和检测的方法，同时训练两个数据集(WordTree)，使得系统能够检测超过9000类的物体；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558516206/github_image/2019-05-22/YOLOv2_WordTree.png)

---

---

### YOLO v3

&emsp; 如下图所示，下图表示YOLO v3的网络结构图；**这里声明一点：图像来源于[木盏-yolo系列之yolo v3【深度解析】](<https://blog.csdn.net/leviopku/article/details/82660381>)，这里介绍YOLO v3的思路也来源于上面的链接；**；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558531017/github_image/2019-05-22/YOLOv3.jpg)

上图中：

**DBL: ** 表是conv2D+BN+leaky，是YOLO v3的基本组件；

**resn: **表示这个res_block里含有n个res_unit，而res_unit是一个残差块；

**concat:** 表示拼接；将darknet中间层和后面某一层的上采样进行拼接；这里是借鉴的FPN的思想；

```python
# YOLO v3层统计
Total: 252
Add: 23
BatchNormalization: 72
Concatenate: 2
Conv2D: 75
InputLayer: 1
LeakyReLU: 72
UpSampling2D: 2
ZeroPadding: 5
```

**Backbone:**

&emsp; 如下图所示，YOLO v3中是没有池化层和全连接层的，v2中的池化层由步长为2的卷积层代替；而在v2，v3中，会经历5次下采样，即5次池化，因此一般都要求输入图像大小是32的倍数，即$416\times 416$；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558533293/github_image/2019-05-22/YOLOv3_backbone.jpg)

YOLO v3的整体结构类似于YOLO v2，但具体细节则存在很大的不同：

**相同点：**

1. 沿用YOLO的“分而治之”的策略，即通过划分单元格来进行检测；
2. 从YOLO v2开始，卷积后面一般跟着BN和leaky ReLU；
3. 端到端的训练，即一个loss function；

**不同点：**

1. v3抛弃最大池化层，使用步长为2的卷积层代替；
2. 引入了残差块；
3. Darknet-53只是整个网络的一部分，后面还有很大一块，其中引入了FPN的思想；

**Output：**

&emsp; 如下图所示，可以看到YOLO v3的输出，有三个tensor，维度分别为$13\times 13\times 255, 26\times 26 \times 255, 52\times 52 \times 255$；这就是借鉴FPN的思想，特征金字塔；得到不同尺度下的特征，并利用该特征金字塔进行目标检测，越精细的特征就可以检测出越精细的物体；

&emsp; YOLO v3中设定一个grid cell中预测3个bbx，而COCO数据集中有80个类别；所以输出尺度才为$S\times S \times3 \times (5 + 80)$；255就是这么来的；这里的计算方式与YOLO v2相似，类别概率由bbx负责，而不是grid cell；

![](https://res.cloudinary.com/chenzhen/image/upload/v1558534165/github_image/2019-05-22/YOLOv3_output.png)

**Some tricks：**

1. **Bounding Box Prediction**

&emsp; 在YOLO v3中，有关bbx的预测类似于YOLO v2；关于bbx的坐标信息，计算方式如v2一致，即网络输出$t_x, t_y, t_w, t_h$，然后根据公式计算对应的$b_x, b_y, b_w, b_h$；而在计算bbx的置信度（在v3中称为Objectness score）时，进行了一些改变；在v2中，是根据下面公式计算objectness score：
$$
\begin{equation}
Pr(Object) * IOU(b, object) = \sigma(t_o)
\end{equation}
$$
其中，$t_o$表示网络输出；

那么在v3中，有这样几条规则来计算objectness score：

* 如果一个bbx与ground truth的IOU最大，则objectness score为1；
* 如果一个bbx与ground truth的IOU不是最大，但是IOU值仍大于一个给定阈值（0.5），仍认为objectness score为1；

（上面这两条有些类似Faster RCNN中过滤anchor box的方式）

&emsp; 还有一点说明的是，文章中指出使用kemeans选择9个anchor prior，这里就对应于上述特征金字塔中的三层特征，每层特征预测三个bbx；

**Loss Function：**

&emsp; 有关v3中的损失，和v1，v2的损失函数大体都是一致的；

&emsp; 另外需要说明的一点是，如果bbx没有Object，在损失中，将不会对坐标或者类别概率计算损失，只计算obejctness score的损失；（其实这一点应该在v1中就有体现了，没有object，还计算bbx的坐标信息和类别概率的损失干嘛啊，置信度confidence，即objectness score就有问题。只不过这里的objectness score的计算方式和之前稍有不同；）

---

---

&emsp; 初步结束，待补充；