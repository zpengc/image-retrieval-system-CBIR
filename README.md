# 基于内容的图像检索系统


## 环境准备

## CMD
```
pip install -r requirements.txt	# 安装依赖
set FLASK_APP=start.py  
set FLASK_ENV=development
flask extract  # 提取图像特征
flask run  # 开启flask开发环境服务器
flask evaluate  # 评估指标
```

## BASH
```
pip install -r requirements.txt	# 安装依赖
export FLASK_APP=start.py
export FLASK_ENV=development
flask extract  # 提取图像特征
flask run  # 开启flask开发环境服务器
flask evaluate  # 评估指标
```

## 数据集

UKBench DataSet [下载链接](https://archive.org/download/ukbench)

论文`Scalable Recognition with a Vocabulary Tree`
公布该数据集

**该项目使用了前2000张图片**

![UKBench Dataset](https://i.ibb.co/PNfXdc3/Examples-of-the-images-in-the-Uk-Bench-dataset-The-datasets-consists-of-groups-of-four-W640.jpg)

该数据集有2550个组，每个组包括4幅相似的图像，当选择一组内的任何一个图像作为输入，结果中最考前的应该是同一组的其他图像。

## 项目结构

## Web界面

```首页```

![](https://i.ibb.co/f8tRN3n/Snipaste-2021-05-12-19-04-29.png)

```结果页```

![](https://i.ibb.co/mJbnm1d/Snipaste-2021-05-20-13-06-23.png)



## 参考文献

- Distinctive image features from scale-invariant keypoints    
- object Recognition from Local Scale-Invariant Features.
- Lost in quantization: Improving particular object retrieval in large scale image databases 
- Hamming embedding and weak geometric consistency for large scale image search 
- Hamming embedding similarity-based image classification
- Three things everyone should know to improve object retrieval 
- Improving bag-of-features for large scale image search
- On the burstiness of visual elements
- Asymmetric hamming embedding: taking the best of our bits for large scale image search 
- Content-based image retrieval and feature extraction: a comprehensive review 
- Bag-of-words representation in image annotation 
- Scalable Recognition with a Vocabulary Tree

## 总结

### 特征构建
利用聚类算法形成Visual Vocabulary，其中每个Visual Word代表一个聚类中心，将图像特征划分到最近的Visual Word，然后形成每幅图像的频率直方图。

算法选择：K-Means算法（基于欧几里得距离）

### kmeans算法
- inertia 惯性：计算所有节点到最近聚类中心的距离的平方和，将所有平方和相加。该指标越小，聚类内部越相似。

- Silhouette Coefficient 轮廓系数：是聚类效果好坏的一种评价方式，它结合内聚度和分离度两种因素。可以用来在相同原始数据的基础上用来评价不同算法。
等于b-a/max(a,b)，启动b等于平均聚类之间距离，a是平均聚类内部距离。

### Hamming Embedding 汉明嵌入算法
computing the similarity between two images based on the distance between their local descriptors

目的：用于解决 codebook size 大小都会导致检索质量较差的问题

在Herve Jegou的论文中，提出了基于Hamming Embedding的相似度比较方法。思路如下：根据图像特征生成二进制签名，然后比较二进制签名的距离。

优点：在K-Means算法中，如果两个特征落到同一个聚类内部，就认为这两个特征匹配成功；根据Hamming Embedding算法，除了要满足同一个聚类的条件外，还需要比较汉明距离，如果小于等于预设阈值（Threshold=24），则认为这两个特征匹配成功。这样就避免了K-Means算法中不合理的K值对于匹配的影响。


### weak geometric consistency 弱几何一致性
使得检索对特征的缩放和旋转更加鲁棒

### Flask
如何开启CSRF protection：https://flask-wtf.readthedocs.io/en/0.15.x/csrf/

如何配置 CSRF的密钥：https://stackoverflow.com/questions/34902378/where-do-i-get-a-secret-key-for-flask

### 构建倒排索引
为了加快检索的速度，我们没有遍历所有特征，而是使用了倒排索引的文件结构。

计算查询图像的特征描述符，利用k-means算法预测出相应的聚类中心，每一个聚类中心保存本聚类内部所有图像特征，包括图像id，二进制标签，几何信息（尺度和角度）。这样，在匹配时候，只需要遍历聚类内部的所有图像特征即可。

![](https://i.ibb.co/NFPb3DG/1.png)

### 重排reRanking
Lowe的论文提出了Lowe’s Ratio Test。我们需要判断Query Image和检索出的每一幅图像的匹配相关性，根据Euclidean Distance，根据Query Image的每一个特征，确定检索图像中距离最短的特征和距离次短的特征。具体处理如下：
If distance1 < distance2 * 0.7 
	then go on
Else 
	discard the match
其中，distance1表示最短的距离，distance2表示次短的距离。该处理可以舍弃90%的错误匹配和5%的成功匹配，但是会增加处理时间。


### 图像算子

#### LOG Laplacian of Gaussian 高斯拉普拉斯算子
> The Laplacian of Gaussian (LoG) operation goes like this. You take an image, and blur it a little. And then, you calculate second order derivatives on it (or, the "laplacian"). This locates edges and corners on the image. These edges and corners are good for finding keypoints.

> But the second order derivative is extremely sensitive to noise. The blur smoothes it out the noise and stabilizes the second order derivative.

> The problem is, calculating all those second order derivatives is computationally intensive. So we cheat a bit.

#### DOG Difference of Gaussian 高斯差分算子
如下图所示，在同一个octave中，我们计算相邻图像的差，同一个octave相邻图像的模糊程度不同。
![](https://i.ibb.co/N3cpnns/Snipaste-2022-02-23-16-38-20.png)

#### blob
> BLOB stand for Binary Large Objects. Well it is used to represent a group of pixels having similar values for intensity but different from the ones surrounding it.
>
> BLOB in an can be detected with the help of techniques like DoG, LoG and Determinant of Hessian.


## SIFT算法
sift特征不受图像尺度、观察角度、光照、旋转方向的影响。

### step1: 构建尺度空间，因为sift是尺度不变的
![](https://i.ibb.co/1T9DSS0/Snipaste-2022-02-23-16-27-58.png)
相同大小的图像组成一个octave，第二个octave图像大小是第一个octave图像大小的一半，以此类推。同一个octave内的图像，大小相同，但是逐渐模糊。

对于sift算法而言，octave的个数和每个octave内的图像个数是两个重要的参数，作者在paper中建议分别设置为4和5，如上图所示。

> In the first step of SIFT, you generate several octaves of the original image. Each octave's image size is half the previous one. Within an octave, images are progressively blurred using the Gaussian Blur operator.

### step2：计算DOG
![](https://i.ibb.co/HP4b68Z/Snipaste-2022-02-23-16-42-34.png)

### 找到关键点
- 找到局部最大值、最小值像素点
如下图所示，x表示当前像素，如果该像素比周围8个像素点大，同时比上下9+9=18个像素点大，则该像素就是局部最大值像素点，局部最小值像素点同理。
![](https://i.ibb.co/Fb6CxX5/Snipaste-2022-02-23-16-45-05.png)

- 找到局部最大值 、最小值子像素点。
> 计算机显示的最小单位是pixel，但是为什么会有subpixel的概念呢，原因是虽然最小的单位是pixel，但是pixel其实仍然是有面积的，因此我们可以先得到subpixel的值，然后再求pixel，这样图像比较准确。

上面求出的局部极值像素点并不是最终的关键点，真正的关键点在这些局部极值像素点的中间区域。
如下图所示：
![](https://i.ibb.co/DLyg0FT/Snipaste-2022-02-23-16-50-07.png)
红色都是第一步求出的局部最大值点、局部最小值点，但是真正的关键点在绿色区域。

![](https://i.ibb.co/H4TnQ2k/Snipaste-2022-02-23-16-54-17.png)
> The author of SIFT recommends generating two such extrema images. So, you need exactly 4 DoG images. To generate 4 DoG images, you need 5 Gaussian blurred images. Hence the 5 level of blurs in each octave.
>
> In the image, I've shown just one octave. This is done for all octaves. Also, this image just shows the first part of keypoint detection. The Taylor series part has been skipped.

**注意**：每个图像的关键点个数都不同，有的是几千，有的是几百，因此，每个图像计算出的特征描述符的个数也不相同

### 关键点方向

### 计算特征描述符


## 资料
- https://littletomatodonkey.github.io/2018/12/07/2018-12-07-Hamming%20Ebedding%E8%AE%B2%E8%A7%A3/
- http://www.cse.psu.edu/~rtc12/CSE486/lecture11.pdf
- https://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/


## 评价指标
mean AP
