> **思路：**如果一个样本在特征空间中的k个最相近的样本中大多数属于某个类别，则该样本也属于该类别；

这段话中涉及到KNN的三要素：**K、距离度量、决策规则**

- **K：**KNN的算法的结果很大程度取决于K值的选择；

![](https://res.cloudinary.com/chenzhen/image/upload/v1553043588/github_image/2019-03-20/1.jpg)

> If it's too small, the we gain efficiency but become susceptible to noise and outlier data points.
>
> If it's too large, the we are at risk of over-smoothing our classification results and increasing bias.

- **距离度量：**

欧式距离：
$$
d(x,y)=\sqrt{\sum_{k=1}^{n}(x_k-y_k)^2}
$$
曼哈顿距离：
$$
d(x,y)=\sqrt{\sum_{k=1}^{n}|x_k-y_k|}
$$

- **决策规则：**
  - **空间中距离越近的点属于一类的可能性就越大；**
  - 使用前K个样本中多数类别；



**算法过程描述：**

1. 计算测试样本与训练数据之间的距离；
2. 得到距离按照递增关系排序；
3. 选取距离最小的K个样本；
4. 确定选择的K个样本中类别出现的频率；
5. 选择出现频率最大的类别作为测试样本的类别；



**KNN算法的优缺点：**
**优点：**

- 精度高，对异常值不敏感、无数据输入假定；
- 简单有效，易理解；

**缺点：**

- 计算复杂度高、空间复杂度高；
- 无法给出任何数据的基础结构信息；

> 训练简单，测试难；
>
> k-NN更适合低维特征空间，而不适合图片，图片是高维特征空间；
>
> 不能进行学习，也就说如果犯错误了，不能变得更聪明；
>
> 仅仅依赖于n维空间的distance做分类



**KNN算法实现（Python）:**

默认：已经获取到训练数据$dataSet \in R^{m \times n}$、训练集标签$labels \in R^n$、测试样本$inX \in R^{1 \times n}$；

```python
def classify(inX, dataSet, labels, k=3):
    # 首先计算测试样本与训练数据之间的欧式距离
    m = dataSet.shape[0]
    diffMat = np.tile(inX, (m, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(dim=1)
    distance = sqDistance ** 0.5
    
    # 排序，并选择距离最近的前k中样本中类别数目最多的类别
    sortedDistanceIndex = distance.argsort()
    classCount = {} # 保存距离最近的前k个样本中的类别个数
    for i in range(k):
        votelabel = labels[sortedDistanceIndex[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1  # 有则返回，无则置0
    sortedClassCount = np.sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]
    
```

**sklearn中使用KNN：**

```python
# *- coding: utf-8 -*

"""
使用sklearn实现knn
"""
# =====================================

from sklearn import neighbors
from sklearn import datasets

# load dataset
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

# model
knn = neighbors.KNeighborsClassifier()

# 训练
knn.fit(iris_x, iris_y)

# 预测
predictedLabel = knn.predict(iris_x)

error = sum(abs(predictedLabel - iris_y)) / len(iris_y)
print(error)
```



### reference

- https://www.cnblogs.com/ybjourney/p/4702562.html
- https://www.cnblogs.com/python-frog/p/8731080.html