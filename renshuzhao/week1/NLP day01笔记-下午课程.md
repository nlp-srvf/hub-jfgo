# 机器学习简介
## 1找规律
机器学习的**本质**就是**找规律**。

### 1.1从找规律说起
y = f(x) 使得 yi = f(xi)

X:（1,2）   （10,20）   （34,65）   （79,43）

Y:     3              30                99             122

Y = ∑xi    或    Y = W * X, W=[1,1]        

![image](https://cdn.nlark.com/yuque/__latex/9b647a3b5d9ec9898e8ae622be676632.svg)

X: (1,  2)     (10,  20)      (34, 65)      (79,43)

Y: (3,-1)     (30,-10)      (99,-31)     (122,36)

Y=w*X ,  w = ![image](https://cdn.nlark.com/yuque/__latex/b979a77ee6d1023855d3b5df6c155330.svg)

![image](https://cdn.nlark.com/yuque/__latex/e473d78d9e588c8284cebf173858beb5.svg)

Y也可以是一个矩阵

X可以是多个矩阵

Y可以是多个矩阵

X可以是不同维度的多个向量

Y可以是不同维度的多个向量

X可以是多个不同维度矩阵



### 1.2 机器学习
#### 1.2.1 机器学习应用
很多时候，我们有数据，希望找到规律，但规律很复杂，所以希望靠机器来挖掘规律

知道花朵的大小、颜色等信息，来判断花的种类

知道身体血压、血脂等指标，来预测是否患病

知道房屋的大小、位置等信息，来预测房价

知道企业的业务、规模等信息，来预测股价知道国家的人口、经济发展等信息，预测未来GDP

#### 1.2.2有监督学习
有监督学习的核心目标建立一个模型（函数），来描述输入（X）和输出（Y）之间的映射关系，价值在于对于新的输入，通过模型给出预测的输出。

![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760501320824-0c431645-c555-4b4e-ac41-b8afb36352f6.png)

监督学习的要点：

1.需要有一定数量的训练样本

2.输入和输出之间有关联关系

3.输入和输出可以数值化表示

4.任务需要有预测价值

有监督学习在人工智能中的应用有文本分类任务、机器翻译任务、图像识别任务和语言识别任务。

#### 1.2.3无监督学习
给予机器的数据没有标注信息，通过算法对数据进行一定的自动分析处理，得到一些结论.

常见任务有聚类、降维、找特征值等等

#### 1.2.4 有监督学习和无监督学习对比
<font style="color:rgba(0, 0, 0, 0.9);">“有监督给‘答案’学映射，无监督没‘答案’找结构。”</font>

| **<font style="color:#000000;">有监督学习与无监督学习对比表</font>** | | |
| :---: | --- | --- |
| **<font style="color:#000000;">维度</font>** | **<font style="color:#000000;">有监督学习 Supervised Learning</font>** | **<font style="color:#000000;">无监督学习 Unsupervised Learning</font>** |
| <font style="color:#000000;">输入数据</font> | <font style="color:#000000;">带标签数据 (X, y)</font> | <font style="color:#000000;">只有特征 X，没有标签</font> |
| <font style="color:#000000;">学习目标</font> | <font style="color:#000000;">学一个 X → y 的映射，预测/分类</font> | <font style="color:#000000;">发现数据内在结构或分布</font> |
| <font style="color:#000000;">典型任务</font> | <font style="color:#000000;">分类、回归</font> | <font style="color:#000000;">聚类、降维、密度估计、异常检测</font> |
| <font style="color:#000000;">常见算法</font> | <font style="color:#000000;">逻辑回归、SVM、随机森林、CNN、BERT 微调</font> | <font style="color:#000000;">K-means、DBSCAN、PCA、t-SNE、Autoencoder、GMM</font> |
| <font style="color:#000000;">评价方式</font> | <font style="color:#000000;">准确率、F1、MSE 等（有 ground-truth）</font> | <font style="color:#000000;">轮廓系数、重构误差、人工可视化（无绝对标准）</font> |
| <font style="color:#000000;">数据成本</font> | <font style="color:#000000;">高：需人工标注</font> | <font style="color:#000000;">低：直接利用原始数据</font> |
| <font style="color:#000000;">举例</font> | <font style="color:#000000;">给 1 万张“猫/狗”图片+标签，训练后识别新图片</font> | <font style="color:#000000;">给 1 万张无标签用户画像，自动分成 5 个簇做精准营销</font> |




#### 1.2.5 强化学习
<font style="color:rgba(0, 0, 0, 0.9);">强化学习（Reinforcement Learning, RL）  
</font>**<font style="color:rgba(0, 0, 0, 0.9);">定义</font>**<font style="color:rgba(0, 0, 0, 0.9);">：智能体在与环境的连续交互中，通过“试错”获得延迟奖励信号，学习一个策略 π，使得长期累积奖励最大化。</font>

<font style="color:rgba(0, 0, 0, 0.9);">四大核心要素：</font>

    - <font style="color:rgba(0, 0, 0, 0.9);">状态 S：环境当前局面</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">动作 A：智能体可执行的操作</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">奖励 R：环境对动作的即时反馈</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">策略 π：从状态到动作的映射，目标是最大化累积折扣奖励 E[Σγ^t R_t]</font>

<font style="color:rgba(0, 0, 0, 0.9);">与 SL/UL 的本质区别：  
</font><font style="color:rgba(0, 0, 0, 0.9);">没有正确标签，只有“好坏评价”；反馈延迟，需探索与利用平衡。</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">1.2.6 强化学习和有监督学习、无监督学习对比</font>
**<font style="color:rgba(0, 0, 0, 0.9);">SL=每题给答案，UL=无答案自己找规律，RL=考完只给总分自己猜错因。</font>**

| **<font style="color:#000000;">机器学习类型对比表</font>** | | | |
| :---: | --- | --- | --- |
| **<font style="color:#000000;">维度</font>** | **<font style="color:#000000;">有监督学习 Supervised</font>** | **<font style="color:#000000;">无监督学习 Unsupervised</font>** | **<font style="color:#000000;">强化学习 Reinforcement</font>** |
| <font style="color:#000000;">数据形式</font> | <font style="color:#000000;">(样本, 标签) 成对出现</font> | <font style="color:#000000;">只有样本，无标签</font> | <font style="color:#000000;">无固定标签，只有环境反馈的“奖励”</font> |
| <font style="color:#000000;">信号时效</font> | <font style="color:#000000;">立即、逐样本</font> | <font style="color:#000000;">无信号</font> | <font style="color:#000000;">延迟、稀疏、对整条轨迹</font> |
| <font style="color:#000000;">优化目标</font> | <font style="color:#000000;">最小化预测与标签误差</font> | <font style="color:#000000;">发现隐藏结构或分布</font> | <font style="color:#000000;">最大化长期累积奖励</font> |
| <font style="color:#000000;">典型任务</font> | <font style="color:#000000;">分类、回归</font> | <font style="color:#000000;">聚类、降维、密度估计</font> | <font style="color:#000000;">决策、控制、游戏、自动驾驶</font> |
| <font style="color:#000000;">代表算法</font> | <font style="color:#000000;">逻辑回归、CNN、BERT</font> | <font style="color:#000000;">K-means、PCA、GAN</font> | <font style="color:#000000;">Q-Learning、DQN、PPO、A3C</font> |
| <font style="color:#000000;">老师类比</font> | <font style="color:#000000;">老师逐题给标准答案</font> | <font style="color:#000000;">没老师，学生自己总结笔记</font> | <font style="color:#000000;">老师期末只给总分，学生自学哪题错</font> |




<font style="color:rgba(0, 0, 0, 0.9);">互补关系</font>  
<font style="color:rgba(0, 0, 0, 0.9);">现实系统常组合：</font>  
<font style="color:rgba(0, 0, 0, 0.9);">SL 预训练 → UL 提取特征 → RL 在线微调，三位一体。</font>

<font style="color:rgba(0, 0, 0, 0.9);">有监督 SL, 无监督 UL, 强化学习 RL，这是按“训练信号”划分的三大范式，</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">1.2.6机器学习的流程</font>
![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760502880943-c4e396a4-492c-43c3-b480-46c799d3ed61.png)

#### 1.2.7 常用概念
1.训练集 : 用于模型训练的训练数据集合

2.验证集 : 对于每种任务一般都有多种算法可以选择，一般会使用验证集验证用于对比不同算法的效果差异

3.测试集 : 最终用于评判算法模型效果的数据集合

4.K折交叉验证（K fold cross validation）:

初始采样分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K-1个样本用来训练。交叉验证重复K次，每个子样本验证一次，平均K次的结果



![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760503551239-165bddfc-8da9-41f5-bc8e-03fca9bb6ff5.png)

5..过拟合 ：模型失去了泛化能力。如果模型在训练集和验证集上都有很好的表现，但在测试集上表现很差，一般认为是发生了过拟合

6.欠拟合 ：模型没能建立起合理的输入输出之间的映射。当输入训练集中的样本时，预测结果与标注结果依然相差很大



**造成过拟合的原因一般是训练数据不足，欠拟合的原因一般是训练模型不匹配。**

![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760516737774-b0ce4c3e-aa9a-4216-9659-0c20bfa918c3.png)

<font style="color:rgb(77, 77, 77);">左图，通过几个数据点的一条直线与真实抛物线欠拟合。</font>

<font style="color:rgb(77, 77, 77);">右图，一条曲线穿过每个数据点与真实抛物线过拟合。</font>

+ **<font style="color:rgb(51, 51, 51);">模型效果太差：</font>**<font style="color:rgb(51, 51, 51);">欠拟合</font>
+ **<font style="color:rgb(51, 51, 51);">模型在训练集上还可以，但测试集上太差：</font>**<font style="color:rgb(51, 51, 51);">过拟合</font>
+ **<font style="color:rgb(51, 51, 51);">模型训练集和测试集都还行：</font>**<font style="color:rgb(51, 51, 51);">不存在过拟合与欠拟合</font>

7.评价指标

为了评价算法效果的好坏，需要找到一种评价模型效果的计算指标。不同的任务会使用不同的评价指标。

常用评价指标：

准确率 召回率 F1值 TOPK BLEU

| **<font style="color:#000000;">指标含义及相关信息表</font>** | | | | |
| --- | --- | --- | --- | --- |
| **<font style="color:#000000;">指标</font>** | **<font style="color:#000000;">人话版含义</font>** | **<font style="color:#000000;">核心公式/解释</font>** | **<font style="color:#000000;">典型场景</font>** | **<font style="color:#000000;">一句话记忆</font>** |
| <font style="color:#000000;">准确率 Accuracy</font> | <font style="color:#000000;">猜对的占总样本的比例</font> | <font style="color:#000000;">(TP+TN)/(TP+TN+FP+FN)</font> | <font style="color:#000000;">类别均衡的分类任务</font> | <font style="color:#000000;">“一刀切看整体”</font> |
| <font style="color:#000000;">召回率 Recall 又叫查全率</font> | <font style="color:#000000;">正例里被你找回了多少</font> | <font style="color:#000000;">TP/(TP+FN)</font> | <font style="color:#000000;">漏掉代价大的场景：医疗诊断、故障检测</font> | <font style="color:#000000;">“宁可错杀，不可漏网”</font> |
| <font style="color:#000000;">F1 值</font> | <font style="color:#000000;">准确率和召回率的调和平均，兼顾两者</font> | <font style="color:#000000;">2·Precision·Recall/(Precision+Recall)</font> | <font style="color:#000000;">类别不平衡、需要综合性能</font> | <font style="color:#000000;">“拉架”指标，谁低拉谁</font> |
| <font style="color:#000000;">Top-K 准确率</font> | <font style="color:#000000;">真实标签落在模型前 K 个最高概率里就算对</font> | <font style="color:#000000;">命中@K / 总样本</font> | <font style="color:#000000;">推荐、图像检索、多标签分类</font> | <font style="color:#000000;">“给 K 次机会”</font> |
| <font style="color:#000000;">BLEU</font> | <font style="color:#000000;">机器译文与参考译文的 n-gram 重合度（0~1）</font> | <font style="color:#000000;">BP·exp(∑w\_n·log p\_n)</font> | <font style="color:#000000;">机器翻译、文本生成</font> | <font style="color:#000000;">“n-gram 精确率套餐”，越高越像人</font> |


## 1.3 深度学习
深度学习(Deep Learning)特指基于深层神经网络模型和方法的机器学习。它是在统计机器学习、人工神经网络等算法模型基础上，结合当代大数据和大算力的发展而发展出来的。深度学习最重要的技术特点是具有自动提取特征的能力，所提取的特征也称为深度特征或深度特征表示，相比于人工设计的特征，深度特征的表示能力更强、更稳健。因此，深度学习的本质是特征表征学习。深层神经网络是深度学习能够自动提取特征的模型基础，深层神经网络本质上是一系列非线性变换的嵌套。目前看来，深度学习是解决强人工智能这一重大科技问题的最具潜力的技术途径，也是当前计算机、大数据科学和人工智能领域的研究热点。

深度学习是一个跨学科的技术领域，涉及数据科学、统计学、工程科学、人工智能和神经生物学，是机器学习的一个重要分支。

<font style="color:rgba(0, 0, 0, 0.9);">深度学习是用多层可微分网络（≥1 个隐藏层）自动学习特征表示，并通过反向传播端到端训练模型的机器学习分支。</font>

### 1.3.1 猜数字
A: 我现在心里想了一个0-100之间的整数，你猜一下？

B: 60。A: 低了。

B：80。A：低了。

B：90。A：高了

B：88。A：对了！

假设A先生选取数字是有规律的，与他选取范围的上限有关

构建模型预测A心里想的数字

模型输入：A给出的上限数值

模型输出：A心里想的数值

#### 1.3.2 深度学习模型-**<font style="color:rgba(0, 0, 0, 0.9);">零阶优化</font>**  
**<font style="color:rgba(0, 0, 0, 0.9);">	没有导数，只靠‘左/右’口令，用试探法把随机初值逐步挪到零误差，是 RL/零阶优化的最简素描。</font>**
首先B随便猜一个数                                           	----模型随机初始化                              

模型函数 ：Y = k * x   (此样本x = 100)                                          

此例子中B选择的初始k值为0.6 

A计算B的猜测与真正答案的差距                 	----计算loss                                          

损失函数 = sign(y_true – y_pred)

A告诉B偏大或偏小                                            	----得到loss值

B调整了自己的“模型参数”                              	----反向传播

参数调整幅度依照B自定的策略                    	----优化器&学习率

重复以上过程

最终B的猜测与A的答案一致                          	----loss = 0

#### 1.3.3 深度学习-优化方法
##### 1.随机初始化。
假如B一开始选择的k值为88，则直接loss=0

NLP中的预训练模型实际上就是对随机初始化的技术优化

隐含层中会含有很多的权重矩阵，这些矩阵需要有初始值，才能进行运算初始值的选取会影响最终的结果一般情况下，模型会采取随机初始化，但参数会在一定范围内在使用预训练模型一类的策略时，随机初始值被训练好的参数代替

##### 2.优化损失函数
假如损失函数为loss = y_true – y_pred

即当B猜测60的时候，A告知低了28

损失函数（loss function或cost function）用来计算模型的预测值与真实值之间的误差。模型训练的目标一般是依靠训练数据来调整模型参数，使得损失函数到达最小值。损失函数有很多，选择合理的损失函数是模型训练的必要条件。

#导数与梯度#导数表示函数曲线上的切线斜率。 除了切线的斜率，导数还表示函数在该点的变化率。

![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760583292398-d1e7a737-a501-4be8-97cd-d9d0be06a837.png)

**导数与梯度**

**导数：**

导数为正数 x变大 y随之变大

导数为负数 x变大 y随之变小

导数的应用：

模型训练就是通过调整输入数据寻找loss变小的过程

**梯度**

梯度告诉我们函数向哪个方向增长最快，那么他的反方向，就是下降最快的方向

梯度下降的目的是找到函数的极小值

为什么要找到函数的极小值？   

因为我们最终的目标是损失函数值最小



损失函数决定调整方向，优化器决定调整幅度

##### 3.调整参数的策略
B采取二分法调整，50 -> 75 -> 88

知道走的方向，还需要知道走多远

**优化器**

假如一步走太大，就可能错过最小值，如果一步走太小，又可能困在某个局部低点无法离开

学习率（learning rate），动量（Momentum）都是优化器相关的概念

![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760583593101-567d4168-80c0-42ba-9885-ec9c0b4b5195.png)

步长不合适有可能进入局部最低点。



    - **<font style="color:rgba(0, 0, 0, 0.9);">Mini-Batch</font>**<font style="color:rgba(0, 0, 0, 0.9);">：把一次 Epoch 再切成若干小批，每批样本数 = batch_size，每跑完一个 Mini-Batch 就更新一次参数（SGD/Adam 等优化器动作）。</font>

一次训练数据集的一小部分，而不是整个训练集，或单条数据

它可以使内存较小、不能同时训练整个数据集的电脑也可以训练模型。

它是一个可调节的参数，会对最终结果造成影响不能太大，因为太大了会速度很慢。 也不能太小，太小了以后可能算法永远不会收敛。

    - **<font style="color:rgba(0, 0, 0, 0.9);">Epoch</font>**<font style="color:rgba(0, 0, 0, 0.9);">：完整扫一遍训练集——一个 Epoch = 所有样本都被用过一次。</font>

我们将遍历一次所有样本的行为叫做一个 epoch

<font style="color:rgba(0, 0, 0, 0.9);">Mini-Batch 与 Epoch 都属于训练阶段，调整参数的策略里的数据投喂方式，用来控制“怎么把样本喂给优化器.</font>



##### 4.调整模型结构
不同模型能够拟合不同的数据集

![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760582018040-741666b7-d1bc-47e3-a636-eb770b8c5e3f.png)



#### 1.3.4人工神经网络
人工神经网络（Artificial Neural Networks，简称ANNs），也简称为神经网络（NN）。它是一种模仿动物神经网络行为特征，进行分布式并行信息处理的算法数学模型。

![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760582248076-baeb1748-58e4-47df-ab82-6c1c62003ecd.png)



隐含层/中间层

神经网络模型输入层和输出层之间的部分隐含层可以有不同的结构, RNN/CNN/DNN/LSTM/Transformer等，它们本质上的区别只是不同的运算公式。

![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760582470444-52e90733-ab49-4f54-819d-6eb453cc9df1.png)	

| **<font style="color:#000000;">结构</font>** | **<font style="color:#000000;">核心机制</font>** | **<font style="color:#000000;">擅长处理</font>** | **<font style="color:#000000;">典型应用</font>** |
| :---: | :---: | :---: | :---: |
| <font style="color:#000000;">DNN（深度神经网络）</font> | <font style="color:#000000;">全连接层堆叠</font> | <font style="color:#000000;">表格数据、简单映射</font> | <font style="color:#000000;">房价预测、分类器 backbone</font> |
| <font style="color:#000000;">CNN（卷积神经网络）</font> | <font style="color:#000000;">卷积 + 池化</font> | <font style="color:#000000;">图像/空间局部特征</font> | <font style="color:#000000;">图像识别、医学影像、目标检测</font> |
| <font style="color:#000000;">RNN（循环神经网络）</font> | <font style="color:#000000;">时间递归结构</font> | <font style="color:#000000;">序列建模（短序列）</font> | <font style="color:#000000;">股票预测、早期语音识别</font> |
| <font style="color:#000000;">LSTM/GRU（RNN 升级版）</font> | <font style="color:#000000;">门控机制，记忆单元</font> | <font style="color:#000000;">长序列依赖</font> | <font style="color:#000000;">文本生成、机器翻译、时间序列预测</font> |
| <font style="color:#000000;">Transformer</font> | <font style="color:#000000;">自注意力机制（Self-Attention）</font> | <font style="color:#000000;">长距离依赖+并行计算</font> | <font style="color:#000000;">BERT、GPT、T5、Vision Transformer（ViT）</font> |


    - **<font style="color:rgba(0, 0, 0, 0.9);">CNN</font>**<font style="color:rgba(0, 0, 0, 0.9);"> 提取图像特征（空间）</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">Transformer</font>**<font style="color:rgba(0, 0, 0, 0.9);"> 提取文本特征（序列）</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">融合层</font>**<font style="color:rgba(0, 0, 0, 0.9);">（如 Cross-Attention）将两者结合</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">输出层</font>**<font style="color:rgba(0, 0, 0, 0.9);"> 给出最终预测（如图像描述生成）</font>

隐含层 ≠ 固定结构，它是“可插拔”的特征提取器 ——是 CNN 就看图，是LSTM/Transformer 就读序列，是 DNN 就做通用映射，按需组合，灵活搭积木。

#### <font style="color:rgba(0, 0, 0, 0.9);">1.3.5深度学习训练流程</font>
![](https://cdn.nlark.com/yuque/0/2025/png/40695672/1760585385777-fc37666a-7bc7-406c-b5d9-7e3da9e7f6e9.png)



训练迭代进行要点：模型结构选择、初始化方式、选择损失函数、选择优化器、选择样本质量数量

模型训练好后把参数保存即可用于对新样本的预测





