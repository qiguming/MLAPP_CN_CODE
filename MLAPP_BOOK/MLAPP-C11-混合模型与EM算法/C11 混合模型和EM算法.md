# 11.混合模型与EM算法

## 11.1 隐变量模型

在第10章，我们展示了如何使用图模型来定义高维空间下的联合概率分布。其基本思想是通过在图中的两个节点之间添加边来模拟两个变量之间的相关性。（从技术上来说，图模型所要表达的是条件独立性，但你应该可以了解到这一点）。

另一种可选的方案是，假设两个可观察到的变量之所以相关，是因为它们都由一个共同的隐藏“原因”引起。含有隐藏变量的模型被称为**隐变量模型**($\rm{latent \ variable \ models, LVMs}$)。正如我们在本章将会看到的，这样的模型比起那些不含隐变量的模型更难训练。然而，它们却十分重要，主要基于两个原因：首先，$\rm{LVMs}$比起那些在可见空间内直接表达相关性的模型具有更少的参数。这一点在图11.1中有所说明。如果所有的节点（包括$H$）是二值变量，并且所有的$\rm{CPDs}$(条件概率分布)是一张概率分布表，那么模型共有$\rm{17}$个自由参数。然而，右边的模型却有$\rm{59}$个自由参数。

其次，在$\rm{LVM}$中，隐变量可以充当一个**瓶颈**（$\rm{bottleneck}$），它给出了数据在压缩后的表达形式。正如我们将会看到的，它构成了无监督学习的基础，图11.2说明了一些适用于这个场景的一般的$\rm{LVM}$结构。在一般情况下，存在$L$个隐变量$z_{i1},...,z_{iL}$和$D$个可见变量$x_{i1},...,x_{iD}$，通常情况下$D \gg L$。如果$L \gt 1$,即存在多个隐变量对每个观测值有贡献，这种情况下的映射为**多对多**。如果$L = 1$，即我们只有一个隐变量，在这种情况下，$z_i$通常是离散的，此时的映射为**一对多**。我们也可以有一个**多对一**的映射，意味着对于每个观测变量而言，存在许多竞争因素，这样的模型是**概率矩阵分解**$(\rm{probabilistic \ matrix \ factorization})$的基础，这一点将在27.6.2节介绍。最后，我们也可以有一个**一对一**的映射，它可以表示为$\mathbf{z}_i \rightarrow \mathbf{x}_i$。如果我们允许$\mathbf{z}_i$ 和（或）$\mathbf{x}_i$是一个向量，那么这种表达方式将涵盖其他所有的形式。根据似然函数$p(\mathbf{x}_i|\mathbf{z}_i)$和先验分布$p(\mathbf{z}_i)$的形式，我们可以产生各种不同的模型，如表11.1中所展示的。

## 11.2 混合模型

LVM的最简单的形式是当$z_i \in \{1,...,K\}$时，代表一个离散隐变量状态。我们将使用一个离散先验分布$p(z_i)={\rm{Cat}}(\mathbf{\pi})$来表示这一点。对于似然函数，我们使用$p(\mathbf{x}_i|z_i=k)=p_k(\mathbf{x}_i)$，其中$p_k$为第$k$个观测值的**基分布(base distribution)**，这个分布可以是任意的形式。这个模型被称为**混合模型(mixture model)**，因为我们将$K$个基分布按照如下的形式进行了混合：
$$
p(\mathbf{x}_i|\mathbf{\theta})=\sum_{k=1}^K \pi_kp_k(\mathbf{x}_i|\mathbf{\theta}) \tag{11.1}
$$
上式被称为$p_k$的**凸组合(convex combination)**，因为我们使用了加权求和，且**混合权重(mixing weights)**$\pi_k$满足$0 \le \pi_k \le 1$和$\sum_{k=1}^K \pi_k=1$。我们将在下面给出一些例子。

<p><img src=".\pic\11.1.png" title="图11.1 一个包含和不包含隐藏变量的DGM。叶子节点代表医学症状。这些根节点代表了主要原因，比如吸烟、节食和锻炼。隐藏变量可以表示中间因素，如心脏病，这可能不是直接可见的."></p>
![](./pic/11.2.png "图11.2 用DGM表示的潜在变量模型。(a)多对多。(b)一对多。(c)多对一。(d)一对一")



|      |      |      |      |
| ---- | ---- | ---- | ---- |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |
|      |      |      |      |



### 11.2.1 高斯混合模型

混合模型中使用最广泛是**高斯混合**$\rm{(mixture \ of  \ Gaussians, MOG)}$,又被称为**高斯混合模型**$\rm{(Gaussian \ mixture  \ model,GMM)}$。在这个模型中，参与混合的每一个基分布是一个多变量高斯分布，其期望值为$\mathbf{\mu}_k$，协方差矩阵为$\mathbf{\sum}_k$。所以这个模型具有如下的形式
$$
p(\mathbf{x}_i|\mathbf{\theta})=\sum_{k=1}^K \pi_k\mathcal{N}(\mathbf{x}_i|\mathbf{\mu}_k,\mathbf{\Sigma_k}) \tag{11.2}
$$
图11.3展示了在2维空间下3个高斯分布的混合。每一个混合的基分布表示为一系列椭圆曲线的集合。如果有足够多的混合元素，那么一个$\rm{GMM}$可以用来近似任意定义在$\mathbb{R}^D$上的概率分布。

![](.\pic\11.3_a.png)

![](.\pic\11.3_b.png)

图11.3 在2维空间下3个高斯分布的混合。(a)每个高斯基分布的等高线；（b)加权和之后的全局概率分布。图形由程序**mixGaussPlotDemo**生成。

### 11.2.2 混合多项式分布

我们可以利用混合模型来定义任意形式的数据的概率密度。举例来说，假设我们的数据由$D$维二值向量构成。在这种情况下，一个近似的类条件密度为多个伯努利分布的乘积：
$$
p(\mathbf{x}_i|z_i=k,\mathbf{\theta})=\prod_{j=1}^D {\rm{Ber}}(x_{ij}|\mu_{jk}) = \prod_{j=1}^D \mu_{jk}^{x_{ij}}(1-\mu_{jk})^{1-x_{ij}} \tag{11.3}
$$
其中$\mu_{jk}$为在簇$k$中第$j$个属性为真的概率。

隐变量本身没有任何意义，我们引入隐变量的目的可能仅仅是为了使模型更具有表达力。举例来说，我们可以发现混合分布的期望和协方差由下式给定
$$
\begin{align}
\mathbb{E}[\mathbf{x}] &=\sum_k\pi_k\mathbf{\mu}_k \tag{11.4} \\
\rm{cov}[\mathbf{x}] &= \sum_k\pi_k[\mathbf{\Sigma}_k+\mathbf{\mu}_k\mathbf{\mu}_k^T] - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T \tag{11.5}\\ 
\end{align}
$$
其中$\mathbf{\sum}_k=diag(\mu_{jk}(1-\mu_{jk}))$。所以尽管参与混合的基分布的形式是因式分解的，但其联合分布并非如此。所以混合分布可以对变量之间的相关性进行建模，而一个单独的“伯努利乘积”模型并不能实现这一点。

(**译者注**：关于式11.4和11.5的证明，以及相关论点的再论述：)

考虑一个高斯混合模型，其中包含$K$个基本高斯分布，混合密度表示为：
$$
p(\mathbf{x})=\sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)
$$
我们可以证明：
$$
\begin{align}
\mathbb{E}[\mathbf{x}] & = \int\mathbf{x}p(\mathbf{x})d\mathbf{x} \\
& = \int \mathbf{x}\sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)d\mathbf{x} \\
& = \sum_{k=1}^K\pi_k\int\mathbf{x}\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)d\mathbf{x} \\
& = \sum_k\pi_k\mathbf{\mu}_k
\end{align}
$$

$$
\begin{align}
{\rm{cov}}[\mathbf{x}] & = \mathbb{E}[\mathbf{x}\mathbf{x}^T]-\mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T \\
& = \int \mathbf{x}\mathbf{x}^T \sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)d\mathbf{x} - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T \\
& = \sum_{k=1}^{K}\pi_k  \int \mathbf{x}\mathbf{x}^T\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)d\mathbf{x} - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T \\
& = \sum_{k=1}^{K}\pi_k [\Sigma_k+\mathbf{\mu}_k\mathbf{\mu}_k^T]-\mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T
\end{align}
$$

根据上面的结论我们可以作如下分析，如果对于某一个基分布$k$而言，其协方差矩阵可能是对角矩阵，也就是说不同维度的变量是互不相关的，也就是说这种分布没有办法单独完成对变量之间的相关性进行建模。但是通过多个这种基分布的混合，却可以对变量之间的相关性进行建模，这一点主要体现在协方差矩阵不再是一个对角矩阵。所以通过混合几种比较基本的模型，我们可以扩展模型的表达能力。

### 11.2.3 使用混合模型进行聚类

混合模型具有两个重要的应用。首先是将它们作为一个**黑箱**$\rm{(black—box)}$密度模型$p(\mathbf{x}_i)$。这一点在很多任务中都有应用，比如数据压缩，异常值检测，以及构造生成式分类器，也就是说我们通过一个混合分布来对每一个条件概率分布$p(\mathbf{x}|y=c)$进行建模（见14.7.3节）。

其次，也是最常用的，是将它们用于聚类。我们将在第25章详细讨论这个话题，其基本思想很简单。我们首先训练一个混合模型，然后计算$p(z_i=k|\mathbf{x}_i,\mathbf{\theta})$，代表点$i$属于簇$k$的后验概率。这被称为簇$k$对点$i$的**责任**$\rm{(responsibility)}$，可以通过贝叶斯法则进行计算：
$$
r_{ik} = p(z_i=k|\mathbf{x}_i,\mathbf{\theta})=\frac{p(z_i=k|\mathbf{\theta})p(\mathbf{x}_i|z_i=k,\mathbf{\theta})}{\sum_{k^\prime}^Kp(z_i=k^\prime|\mathbf{\theta})p(\mathbf{x}_i|z_i=k^\prime,\mathbf{\theta})} \tag{11.6}
$$
上式被称为**柔和聚类（soft clustering）**,其计算量与使用生成式分类器的计算量一样。两种模型的区别仅仅在于训练阶段：在混合模型中，变量$z_i$不具备观测值，然而，在生成式模型中，$y_i$（扮演着$z_i$的角色）具有观测值。

我们可以将分类的不确定度表示为$1-\max_kr_{ik}$。假设这个值很小，那么我们使用$\rm{MAP}$估计计算**硬聚簇（hard clustering）**也是合理的。
$$
z_i^*={\rm{arg}}\max \limits_{k} r_{ik}={\rm{arg}}\max \limits_{k} \log p(\mathbf{x}_i|z_i=k,\mathbf{\theta})+\log p(\mathbf{z}_i=k|\mathbf{\theta}) \tag{11.7}
$$
图1.8中给出了使用GMM进行硬聚簇的例子，其中我们将一些表示身高和体重的数据进行聚类，不同颜色代表不同类别。值得注意的是，标签所使用的身份并不重要，我们完全可以对所有的簇进行重新命名，且不影响数据的聚类；这被称为**标签置换（label switching）**。

![](.\pic\11.4_a.png)



![](.\pic\11.4_b.png)

图 11.4  (a) 一些酵母基因的表达水平随时间的变化;(b)利用K-means形成的16个簇的中心。图形由程序    		**kmeansYeastDemo**生成

另一个例子在图11.4中进行展示。其中数据向量$\mathbf{x}_i\in\mathbb{R}^7$表示不同基因在7个不同时间点的表达水平。我们使用$ GMM $进行聚类。发现存在几种基因，比如那些表达水平随着时间单调上升的基因（对给定刺激的响应），那些表达水平单调下降的基因，以及那些呈复杂变化趋势的基因。我们将这些基因聚类成$K=16$类。（11.5节将介绍如何确定$K$值。）举例来说，我们可以将每个簇表示为一个**原型（prototype）**或者**质心（centroid）**。图11.4(b)给出了展示。

![11.5](E:\MLAPP翻译\MLAPP_BOOK\MLAPP-C11-混合模型与EM算法\pic\11.5.png)

图 11.5 针对二值化的手写数字训练一个混合模型，该混合模型由10个伯努利分布组成。我们展示了对应的簇期望$\mathbf{\mu}_k$的MLE。每张图片上方的数字代表了混合权重$\hat{\pi}_k$。在训练的过程中，没有使用标签数据。图形由程序**mixBerMnistEM**生成。

接下来我们将给出一个对二值数据进行聚类的例子，考虑如图1.5(a)所示的MNIST手写数据集的二值化版本，其中我们将类别标签忽略掉。我们可以训练一个混合伯努利分布，其中$K=10$，然后观察每个簇的质心$\mathbf{\mu}_k$，如图11.5所示。我们发现这种方法正确地找到了一些数字类别，但整体上的效果并不好：它为某些数字创造了多个簇，但对其他的数字却没有产生任何簇。之所以存在这些“错误”可能是基于以下几个原因：

- 模型过于简单以至于无法获取一个数字的相关的视觉特征。举例来说，每个像素的生成被看成是彼此独立的，导致模型中并没有类似于形状或者笔画的概念。
- 尽管我们认为应该存在10个簇，但一些数字实际上呈现出相当程度的视觉变化。举例来说，数字7就有两种不同的写法（有横梁和没有横梁）。图1.5(a)说明了书写风格的一些变化范围。所以我们需要$K \gg 10$个簇来充分地对这些数据进行建模。然而，如果我们设置的$K$值过大，那么在模型或者算法中将没有东西去阻止额外的簇被用来为同一个数字创建不同的版本。当然，这也是真实发生的情况。我们可以使用模型选择的方法来阻止许多簇都被选择，但是视觉上看起来吸引人与创建一个好的密度估计器是相当不同的。
- 似然函数是一个非凸函数，我们可能遇到了局部最优解，这一点我们将在11.3.2节解释。

这个例子是混合模型中的典型案例，它表明我们应该很小心地对通过模型生成的簇进行“解释”。（使用少量的监督，或者有信息先验可以起到很大的作用）。

### 11.2.4 混合专家

我们将在14.7.3节介绍在生成式分类器情况下如何使用混合模型，当然我们也可以使用它们构造判别式模型，实现分类和回归。举例来说，考虑图11.6(a)中的数据，看起来一个好的模型应该是3个不同的线性回归函数的组合，每一个函数都对应输入空间的不同部分。为了实现建模，混合权重以及混合密度需要与输入相关：
$$
\begin{align}
p(y_i|\mathbf{x}_i,z_i=k,\mathbf{\theta}) &= \mathcal{N}(y_i|\mathbf{w}_k^T\mathbf{x}_i,\sigma_k^2)\tag{11.8}\\
p(z_i|\mathbf{x}_i,\mathbf{\theta}) &= {\rm{Cat}}(z_i|\mathcal{S}(\mathbf{V}^T\mathbf{x}_i))\tag{11.9}\\
\end{align}
$$
图11.7(a)给出了DGM（有向图模型）的示意图。
$$
Figure11.6
$$

![11.7](.\pic\11.7.png)
$$
Figure11.7   (a)混合专家(b)分层混合专家
$$

这个模型被称为**专家混合（mixture of experts, MoE）**。其思想在于每一个子模块都被认为是一个在输入空间中特定领域的”专家“。函数$p(z_i=k|\mathbf{x}_i,\mathbf{\theta})$被称为**门函数(gating function)**，它根据输入值来决定使用哪个专家。举例来说，图11.6(b)展示了三个“专家”如何“瓜分”1维的输入空间。图11.6(a)展示了每个专家单独的预测结果（在这种情况下，专家就是简单的线性回归模型），通过使用如下公式，我们可以计算出模型的整体预测结果，如图11.6(c)所示：
$$
p(y_i|\mathbf{x}_i,\mathbf{\theta})=\sum_{k}p(z_i=k|\mathbf{x}_i,\mathbf{\theta})p(y_i|\mathbf{x}_i,z_i=k,\mathbf{\theta}) \tag{11.10}
$$
我们将在11.4.3节讨论如何训练这个模型。
$$
Figure 11.8
$$
显然，我们可以将任何模型作为专家。举例来说，我们可以使用神经网络（第16章）来同时表示门函数和专家。结果被称为**混合密度网络（mixture density network）**。这样的模型训练起来更慢，但相较于专家混合模型更加灵活。

我们也可以使用专家混合模型作为一个专家。这种模型被称为**分层专家混合（hierarchical mixture of experts）**。图11.7(b)给出了DGM的示意图，章节16.2.6将讨论更多细节。

#### 11.2.4.1 反向问题的应用

专家混合模型在解决**反向问题（inverse problem）**时十分有用。在这个问题中我们需要将多对一的映射进行反向处理。一个典型的例子是在机器人学中，末梢执行器（机械臂）的位置$\mathbf{y}$是由发动机的关节角$\mathbf{x}$唯一确定的。然而，对于一个给定的位置$\mathbf{y}$，可能存在多种$\mathbf{x}$可以产生这个位置。所以逆映射$\mathbf{x}=f^{-1}(\mathbf{y})$却不是唯一的。另一个例子是视频中人物的**运动追踪(kinematic tracking)**，由于自遮蔽的原因，从图像到位置的映射并不是唯一的。

为了说明反向问题的原理，一个更加简单的例子如图11.8(a)所示。我们发现它定义了一个函数$y=f(x)$，因为对于每一个延水平方向的$x$都对应一个唯一的响应值$y$。这通常被称为**前向模型（forwards model）**。现在考虑计算$x=f^{-1}(y)$的问题。对应的反向模型如图11.8(b)所示，该图只是将(a)中的$x$和$y$轴互换而已。现在我们发现在水平轴上的某些值，存在多个对应的输出，所以反问题无法唯一确定。举例来说，如果$y=0.8$，那$x$可能等于0.2或者0.8。因此，预测的分布$p(x|y,\mathbf{\theta})$是一个复合模型。

对于上述数据，我们可以训练一个线性专家的混合模型。图11.8(b)展示了每个专家的预测结果，图11.8(c)展示了（在点估计下）后验预测的峰值和期望。值得注意的是，后验期望并不是一个好的预测。事实上，任何以最小化均方差为训练目标的模型——哪怕这个模型是一个灵活的非线性模型，比如神经网络——在类似于这样的反问题上，效果都不好。然而，后验分布的峰值因为与输入相关，所以提供了一个合理的近似（**译者注：未理解**）。



Figure 11.9

## 11.3 混合模型中的参数估计

我们已经知道了，在假设参数已知的情况下，如何根据观测到的变量值来计算潜变量的后验分布。本节，我们将讨论如何学习这些参数。

在10.4.2节，我们发现，如果观测到的数据是完整的且先验分布具备因式分解形式，那么参数的后验分布也应该具备因式分解的形式，从而使得计算变得简单。然而，不幸的是，如果存在隐变量和（或者）缺失数据，这个结论就不再成立了。根据图11.9我们很容易发现原因。如果$z_i$的值被观察到，然后通过d分离，我们可以知道$\mathbf{\theta}_z \perp \mathbf{\theta}_x|\mathcal{D}$，所以后验分布也可以进行因式分解。但是，在$LVM$中，$z_i$是隐藏的变量，所以参数将不再独立，后验分布将无法进行因式分解，使其计算更加困难。这也使得MAP和ML估计也变得困难，这一点我们将在后面讨论。

### 11.3.1 不可辨识性

对于LVM而言，计算$p(\mathbf{\theta}|\mathcal{D})$的主要问题在于后验分布可能存在多个峰值。为了说明为什么，考虑一个GMM模型。如果$z_i$的值被全部观察到，参数的后验分布将会是一个单峰的分布：
$$
p(\mathbf{\theta}|\mathcal{D})={\rm{Dir}}(\mathbf{\pi}|\mathcal{D})\prod_{k=1}^K{\rm{NIW}}(\mathbf{\mu}_k,\mathbf{\Sigma}_k|\mathcal{D}) \tag{11.11}
$$
因此我们可以很容易地找到全局最优的MAP估计值（全局最优的MLE也容易找到）。

但是，现在假设$z_i$是隐藏变量。在这种情况下，对于每一种“填充”$z_i$的方式，我们都可以得到一个不同的单峰似然函数。所以如果我们关于$z_i$求和，将得到一个多峰的后验分布$p(\mathbf{\theta}|\mathcal{D})$，这些峰值分别对应不同的簇标签。这在图11.10(b)中给出了说明，在图中我们绘制了似然函数$p(\mathcal{D}|\mu_1,\mu_2)$，其中数据来自于图11.10(a)所示的分布。我们在图中发现了两个峰值，一个对应于$\mu_1=-10,\mu_2=10$，另一个对应于$\mu_1=10,\mu_2=-10$。我们称参数是**不可辨识的(identifiable)**，因为没有一个唯一的$MLE$，所以也没有唯一的$MAP$估计（假设先验分布并没有排除掉某个标签），所以后验分布必然是多峰的。在后验分布中存在多少个峰值是很难回答的。理论上存在$K!$个可能的峰值，但某些峰值彼此合并了。然而，峰值的数目可能是指数级别的，因为找到$GMM$的最优$MLE$是一个$NP$-难的问题。

不可辨识性对于贝叶斯推断也存在一个问题。举例来说，假设我们从后验分布$\mathbf{\theta}^{(s)} \sim p(\mathbf{\theta}|\mathcal{D})$采样得到了一些样本，然后对它们求取均值，尝试估计后验期望$\bar{\mathbf{\theta}}=\frac{1}{S}\sum_{s=1}^S\mathbf{\theta}^{(s)}$（这种蒙特卡洛方法将在第24章介绍更多细节）.如果样本来自于不同的峰值，那么均值将毫无意义。然而需要注意的是，对后验预测分布进行求均值是合理的，即$p(\mathbf{x})\approx \frac{1}{S}\sum_{s=1}^Sp(\mathbf{x}|\mathbf{\theta}^{(s)})$,因为似然函数与参数来自于哪个峰值并无关系。

多种解决参数不可辨识性问题的方法已经被提出来。这些方案与模型的细节和使用的推理算法有关系。举例来说，某文献中使用MCMC方法来解决在混合模型中参数不可辨识性的问题。

我们在本章所使用的方法更加简单：我们只计算一个单独的局部峰值，比如说，我们计算近似的MAP估计。（此处我们说是近似，是因为找到最优的MLE以及MAP是个NP难问题，至少对于混合模型来说是这样的）。因为其简单性，它是迄今为止最经常使用的方法。它是一个合理的近似，至少对于样本量充分大的情况下是这样的。为了说明为什么，考虑图11.9(a)。我们发现存在$N$个潜变量，每一个都可以“看到”一个数据点。然而，却只有2个潜在参数，每一个都可以“看到”$N$个数据点。所以关于参数的后验分布的不确定性要比关于潜变量后验分布的不确定性小的多。这佐证了一个计算$p(z_i|\mathbf{x}_i,\hat{\mathbf{\theta}})$，而不需要计算$p(\mathbf{\theta}|\mathcal{D})$的常规策略。（**译者注**：此处的意思是我们在计算关于$\mathbf{x}_i$的后验概率时，使用的是确定的参数估计，而不再需要考虑参数$\theta$的不确定度，这也解释了在EM算法中，我们假设上一次估计的参数值是固定的，**但此处无法理解的是为什么潜变量的后验分布的不确定度要比潜在参数的大**）在第5.6节，我们讨论了分层贝叶斯模型，其间我们必须为参数赋予某种结构。在这样的模型中，计算$p(\mathbf{\theta}|\mathcal{D})$十分重要，因为只有这样，参数自身之间才能传递信息。如果我们使用点估计，这将不再有可能发生。

### 11.3.2 MAP估计是一个非凸问题

在之前的章节中，我们通过启发式的方式讨论了似然函数具有多个峰值的问题，从而导致MAP以及ML估计是十分困难的。本节，我们将以更加代数的方式来说明这一点，这将让我们对这个问题有额外的深入。

考虑一个LVM的对数似然：
$$
\log p(\mathcal{D}|\mathbf{\theta})= \sum_{i}\log\left[\sum_{\mathbf{z}_i}p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})\right] \tag{11.12}
$$
不幸地是，这个目标函数很难最大化，因为我们并不能将$\log$放进求和中。但这仅仅排除了代数上简化的一些方法，并不能说明解决这个问题是困难的。

现在假设联合概率分布$p(\mathbf{z}_i,\mathbf{x}_i|\mathbf{\theta})$是一个指数族分布，意味着它可以写成：
$$
p(\mathbf{x},\mathbf{z}|\mathbf{\theta})=\frac{1}{Z(\mathbf{\theta})}\exp\left[\mathbf{\theta}^T\mathbf{\phi}(\mathbf{x,z})\right] \tag{11.13}
$$
其中$\mathbf{\phi}(\mathbf{x,z})$为充分统计量，$Z(\mathbf{\theta})$为归一化常数（见9.2节介绍更多细节）。不难发现MVN是一个指数族分布，与我们至今遇到的很多分布一样，其中包括狄利克雷分布，multinomial分布，伽玛分布，威舍特分布等等。（学生分布布是一个典型的例外。）更进一步地，如果混合指示变量（**译者注**:即隐变量）的值被观察到了，那么指数族分布的混合依然属于指数族分布。

基于这个假设，**完全数据的对数似然（complete data log likelihood）**可以写成：
$$
l_c(\mathbf{\theta})=\sum_i\log p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})=\mathbf{\theta}^T(\sum_i\mathbf{\phi}(\mathbf{x}_i,\mathbf{z}_i))-NZ(\mathbf{\theta}) \tag{11.14}
$$
第一项显然与$\mathbf{\theta}$呈线性关系。可以证明$Z(\mathbf{\theta})$是一个凸函数，所以整个目标函数是凹的（考虑到减法），所以存在唯一的最大值。

现在考虑当存在缺失数据时会发生什么。**观测数据对数似然（observed data log likelihood）**由下式给定：
$$
l(\mathbf{\theta})=\sum_i\log \sum_{\mathbf{z}_i}p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})=\sum_i\log\left[\sum_{\mathbf{z}_i}e^{\mathbf{\theta}^T\mathbf{\phi}(\mathbf{z}_i,\mathbf{x}_i)}\right]-N\log Z(\mathbf{\theta}) \tag{11.15}
$$
结果表明log-sum-exp函数是一个凸函数，而$Z(\mathbf{\theta})$也是一个凸函数。然而，两者的差并不一定是凸函数。所以目标函数既不是凸函数也不是凹函数，拥有局部的最优值。

非凸函数的缺点在于通常很难找到它的全局最优解。大部分算法只能找到局部最优解，且最终的结果与优化的起始点相关。存在一些优化算法，比如模拟退火算法（见24.6.1节）或者基因遗传算法，这些算法声称总可以找到全局最优解，但必须基于一个不现实的假设（如果它们被允许以“无穷慢”的速度冷却或者允许“无限长”的运行）。实际上，我们会使用一个局部优化器，可能使用**多重随机重新启动（multiple random restarts）**来增加找到一个好的局部最优解的几率。当然，好的初始化也会起很大的作用。我们将会在后面给出如何实现这一点的案例。

值得注意的是，以一种凸方式对混合高斯模型进行优化的方式已经有人提出了。其思想是，我们为每一个数据点分配一个集群，并使用凸$l_1$型惩罚从中选择一个集群，而不是尝试优化集群中心的位置。 这本质上是稀疏内核逻辑回归中使用的方法的无监督版本，我们将在第14.3.2节中讨论。 但请注意，$l_1$惩罚虽然是凸的，但不一定是促进稀疏性的好方法，如第13章所述。事实上，正如我们将在该章中看到的，一些最好的稀疏性促进方法使用非凸型惩罚，并用EM来优化它们！ 这个故事的寓意是：不要害怕非凸性。（**译者注：**该段目前难以理解）
$$
表11.2
$$


## 11.4 EM算法

在机器学习和统计学的很多模型中，如果我们观察到所有的相关随机变量的值，也就是所谓的完整的数据，那么计算参数ML估计或者MAP估计将会十分容易。然而，如果存在缺失值和（或者）潜在变量，那么计算ML/MAP估计将会变得困难。

一种方法是使用一般的基于梯度的优化器来找到**负对数似然（negative log likelihood,NLL）**的局部最小值：
$$
{\rm{NLL}}(\mathbf{\theta}) \triangleq -\frac{1}{N}\log p(\mathcal{D}|\mathbf{\theta}) \tag{11.16}
$$
然而，我们通常不得不添加一些约束，比如协方差矩阵必须是正定的，混合权重的和必须等于1等等，而这些可能会变得十分麻烦。在这种情况下，使用一种被称为**期望最大化（expectation maximization,EM）**的算法会简单的多。这是一种简单的迭代算法，且通常在每一步进行封闭地更新。更进一步地，这个算法自动地满足上述的一些约束。

EM算法利用了如下的事实：如果数据被全部观察到了，那么ML/MAP估计将会变得十分简单。特别地，EM是一个迭代算法，它在“已知参数推理缺失值（E步骤）”和“给定缺失值后优化参数（M步骤）”之间进行迭代。我们将在下文给出细节以及几个应用该算法的例子。同时在最后给出一个更加理论性的讨论，那时，我们将在更高的层面讨论这个算法。表11.2给出了本书中使用到EM算法的案例。

### 11.4.1 基本思想

令$\mathbf{x}_i$为第$i$个可观测变量，$\mathbf{z}_i$为对应的隐变量。我们的目标是最大化观测数据的对数似然：
$$
l(\mathbf{\theta})=\sum_{i=1}^{N}\log p(\mathbf{x}_i|\mathbf{\theta})=\sum_{i=1}^{N}\log \left[\sum_{\mathbf{z}_i}p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})\right] \tag{11.17}
$$
不幸地是，上式很难进行优化，因为对数$\log$无法放入到求和符号里面。

$\rm{EM}$使用下面的方法解决这个问题。定义**完整数据的对数似然（complete data log likelihood）**为：
$$
l_c(\mathbf{\theta})\triangleq \sum_{i=1}^N \log p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta}) \tag{11.18}
$$
上式无法进行计算，因为$\mathbf{z}_i$是未知的。所以让我们定义**完整数据的对数似然期望（expected complete data log likelihood）**：
$$
Q(\mathbf{\theta},\mathbf{\theta}^{t-1})=\mathbb{E}[l_c(\mathbf{\theta})|\mathcal{D},\mathbf{\theta}^{t-1}] \tag{11.19}
$$
其中$t$表示当前迭代次数。$Q$被称为**辅助函数（auxiliary function）**。期望值是关于旧的参数$\mathbf{\theta}^{t-1}$和观察到的数据$\mathcal{D}$计算得到的。其中$\mathbf{E}$步骤就是为了计算$Q(\mathbf{\theta},\mathbf{\theta}^{t-1})$，或者更准确的说，是求解那些在计算$\rm{MLE}$时所需要的相关项，又被称为**期望充分统计量（expected sufficient statistics，ESS）**。在$\mathbf{M}$阶段，我们关于$\mathbf{\theta}$最大化$Q$函数：
$$
\mathbf{\theta}^t=\arg \max \limits_{\mathbf{\theta}} Q(\mathbf{\theta},\mathbf{\theta}^{t-1}) \tag{11.20}
$$
为了进行$\rm{MAP}$估计，我们可以按照下面的方式对$\rm{M}$步骤进行调整：
$$
\mathbf{\theta}^t=\arg \max \limits_{\mathbf{\theta}}Q(\mathbf{\theta},\mathbf{\theta}^{t-1})+\log p(\mathbf{\theta}) \tag{11.21}
$$
接下来的步骤$\rm{E}$保持不变。

在章节11.4.7，我们将展示EM算法可以单调增加观测数据的对数似然值（如果使用MAP估计，还要加上对数先验），或者保持不变。所以，如果目标函数出现了下降，那一定是我们的计算或者代码出了问题。

接下来我们将展示在几个简单模型中的步骤$\rm{E}$和步骤$\rm{M}$，这样可以让我们对上面的内容更加清晰。

### 11.4.2 GMMs的EM

本节我们将讨论如何使用$\rm{EM}$算法训练一个高斯混合模型。训练其他类型的模型只需要一些直接的修改。我们假设参与混合的基分布数量为$K$，且为已知值（$K$值的选择我们将在11.5节进行讨论）。

#### 11.4.2.1 辅助函数

完整数据的对数似然期望由下式给定：
$$
\begin{align}
Q(\mathbf{\theta},\mathbf{\theta}^{(t-1)}) & \triangleq \mathbb{E}\left[\sum_{i}\log p(\mathbf{x}_i,z_i|\mathbf{\theta})\right] \tag{11.22} \\
& = \sum_i \mathbb{E}\left[\log\left[\prod_{k=1}^K(\pi_kp(\mathbf{x}_i|\mathbf{\theta}_k))^{\mathbb{I}(z_i=k)}\right]\right] \tag{11.23} \\
& = \sum_i\sum_k \mathbb{E}[\mathbb{I}(z_i=k)]\log[\pi_kp(\mathbf{x}_i|\mathbf{\theta}_k)] \tag{11.24} \\
& = \sum_i\sum_k p(z_i=k|\mathbf{x}_i,\mathbf{\theta}^{t-1}) \log[\pi_kp(\mathbf{x}_i|\mathbf{\theta}_k)] \tag{11.25} \\
& = \sum_i\sum_k r_{ik}\log \pi_k + \sum_i\sum_k r_{ik}\log p(\mathbf{x}_i|\mathbf{\theta}_k) \tag{11.26}
\end{align}
$$
其中$r_{ik} \triangleq p(z_i=k|\mathbf{x}_i,\mathbf{\theta}^{(t-1)})$为簇$k$对数据点$i$所负的**责任(responsibility)**。该值在步骤$\rm{E}$进行计算，我们在下面介绍这些内容。

#### 11.4.2.2  $\rm{E}$  step

$\rm{E}$步骤具有如下的简单形式，对于任何混合模型而言，下式都是一样的：
$$
r_{ik}=\frac{\pi_kp(\mathbf{x}_i|\mathbf{\theta}_k^{(t-1)})}{\sum_{k^\prime}\pi_{k^\prime}p(\mathbf{x}_i|\mathbf{\theta}_{k^\prime}^{(t-1)})} \tag{11.27}
$$

#### 11.4.2.3 $\rm{M}$  step

在步骤$\rm{M}$中，我们需要关于参数$\mathbf{\pi}$和$\mathbf{\theta}_k$对辅助函数$\mathcal{Q}$进行优化。对于$\mathbf{\pi}$，我们显然有：
$$
\pi_k=\frac{1}{N}\sum_ir_{ik}=\frac{r_k}{N} \tag{11.28}
$$
其中$r_k \triangleq \sum_ir_{ik}$为所有属于簇$k$的数据点的加权数量。

为了推导出在步骤$\rm{M}$中的$\mathbf{\mu}_k$和$\mathbf{\Sigma}_k$，我们观察在辅助函数$Q$中与$\mathbf{\mu}_k$和$\mathbf{\Sigma}_k$有关的部分。我们发现结果为：
$$
\begin{align}
l(\mathbf{\mu}_k,\mathbf{\Sigma}_k) & = \sum_k\sum_ir_{ik}\log p(\mathbf{x}_i|\mathbf{\theta}_k) \tag{11.29} \\
& = -\frac{1}{2}\sum_ir_{ik}[\log\left|\mathbf{\Sigma}_k\right|+(\mathbf{x}_i-\mathbf{\mu}_k)^T\mathbf{\mathbf{\Sigma}}_k^{-1}(\mathbf{x}_i-\mathbf{\mu}_k)] \tag{11.30}
\end{align}
$$
这只是计算MVN的MLEs的标准问题（见4.1.3节）的加权版本。结果表明，新的参数估计由下式给定：
$$
\begin{align}
\mathbf{\mu}_k & = \frac{\sum_ir_{ik}\mathbf{x}_i}{r_k} \tag{11.31} \\
\mathbf{\Sigma}_k &= \frac{\sum_ir_{ik}(\mathbf{x}_i-\mathbf{\mu}_k)(\mathbf{x}_i-\mathbf{\mu}_k)^T}{r_k}=\frac{\sum_ir_{ik}\mathbf{x}_i\mathbf{x}_i^T}{r_{ik}}-\mathbf{\mu}_k\mathbf{\mu}_k^T \tag{11.32}
\end{align}
$$
这些公式与我们的直觉是一致的：簇$k$的期望是所有属于这个簇的数据点的加权平均，协方差矩阵正比于经验散布矩阵的加权版本。

在计算出新的估计值后，我们令$\mathbf{\theta}^t=(\pi_k,\mu_k,\mathbf{\Sigma}_k),k=1:K$，并且进入步骤$\rm{E}$。

图11.11

#### 11.4.2.4 例子

实际中的一个例子可以参考图11.11。算法的初始值为$\mathbf{\mu}_1=(-1,1),\mathbf{\Sigma}_1=\mathbf{I},\mathbf{\mu}_2=(1,-1),\mathbf{\Sigma}_2=\mathbf{I}$。我们将数据点注上颜色，其中蓝色的点属于簇1，红色的点属于簇2。更加精确地，我们按照下面的方式着色：
$$
{\rm{color}(i)}=r_{i1}{\rm{blue}}+r_{i2}{\rm{red}} \tag{11.33}
$$
所以模糊的点会表现出紫色。在20次迭代后，算法将收敛于一个好的聚簇效果。（我们在进行训练前，通过减去均值，除以标准差，实现了对数据的标准化。这通常有助于收敛。）

#### 11.4.2.5 K-means算法

上述应用在GMMs模型的EM算法存在一个著名的变体：**K-means算法**，也就是我们现在要讨论的算法。考虑一个GMM模型，其中遵循如下的假设：$\mathbf{\Sigma}_k=\sigma^2\mathbf{I}_D$是固定值，$\pi_k=1/K$是固定值，只有聚簇中心$\mathbf{\mu}_k\in\mathbb{R}^D$需要估计。现在考虑将步骤E中进行的后验计算采取一个delta函数近似：
$$
p(z_i=k|\mathbf{x}_i,\mathbf{\theta}) \approx \mathbb{I}(k=z_i^*) \tag{11.34}
$$
其中$z_i^*=\arg \max_{k}p(z_i=k|\mathbf{x}_i,\mathbf{\theta}) $。这也是也被称为**hard EM**，因为我们对所有的点采取一种简单的方式进行分配，既然我们假设每个簇都具有一个相等的球状协方差矩阵，那么对于$\mathbf{x}_i$而言，最有可能属于的那个簇可以通过下式计算得到：
$$
z_i^*=\arg\min_\limits{k}\|\mathbf{x}_i-\mathbf{\mu}_i\|_2^2 \tag{11.35}
$$
所以在每一个步骤E，我们都需要找到$N$个数据点和$K$个聚簇中心的欧几里得距离，其时间复杂度为$O(NKD)$。然而，这个过程可以通过各种方式进行加速，比如应用三角不等式来避免一些冗余的计算。考虑到是hard分配，步骤M更新簇中心的方式只是计算所有属于该簇的数据点的均值：
$$
\mathbf{\mu}_k=\frac{1}{N_k}\sum_{i:z_i=k}\mathbf{x}_i \tag{11.36}
$$
算法11.1给出了伪代码。

| 算法11.1                                                     |
| ------------------------------------------------------------ |
| 1.初始化$\mathbf{\mu}_k$;                                    |
| 2.Do:                                                        |
| 3.      将每个数据点分配给离他最近的中心点：$z_i=\arg \min_k||\mathbf{x}_i-\mathbf{\mu}_k||_2^2$; |
| 4.      根据属于某个簇的所有数据点更新该簇的中心：$\mathbf{\mu}_k=\frac{1}{N_k}\sum_{i:z_i=k}\mathbf{x}_i$; |
| 5.Until converged;                                           |


$$
图11.12
$$

#### 11.4.2.6 矢量量化

K-means并不是一个合适的EM算法，因为它并没有最大化似然函数。取而代之的是，它更接近于最小化一个关于数据压缩的损失函数的贪婪算法，关于这一点我们将在下文进行解释。

假设我们想对一些实值向量$\mathbf{x}_i\in\mathbb{R}^D$进行有损压缩。一种简单的方法是使用**矢量量化(vector quantization, VQ)**。基本思想就是用一个离散符号$z_i\in\{1,...,K\}$代替实值向量$\mathbf{x}_i \in \mathbb{R}^D$，其中离散符号$z_i$代表一个含$K$种原型$\mathbf{\mu}_k \in \mathbb{R}^D$的**码本(codebook)**的索引号。每一个数据向量通过使用最相似的原型的索引号进行编码，其中这里的相似度使用欧式距离来衡量：
$$
{\rm{encode}}(\mathbf{x}_i)=\arg \min_\limits{k}\|\mathbf{x}_i-\mathbf{\mu}_k\|^2 \tag{11.37}
$$
我们可以定义一个损失函数来衡量一个码本的质量，该损失函数定义为**重构损失（reconstruction error）**或者**失真(distortion)**：
$$
J(\mathbf{\mu},\mathbf{z}|K,\mathbf{X})\triangleq \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{x}_i-{\rm{decode}}({\rm{encode}}(\mathbf{x}_i))\|^2=\frac{1}{N}\sum_{i=1}^{N}\|\mathbf{x}_i-\mathbf{\mu}_{z_i}\|^2 \tag{11.38}
$$
其中${\rm{decode}}(k)=\mathbf{\mu}_k$。K-means可以看作是一种简单的通过迭代的方式，实现最小化该损失函数的算法。

当然，当然我们也可以实现零失真，即为每一个数据向量分配一个原型，但是这将占用$O(NDC)$量级的空间，其中$N$表示实数向量的数量，每个向量的长度为$D$,$C$表示存储一个实数标量需要的位数(量化精度)。然而，在很多数据集中，我们会重复的看到相似的向量，所以与其将那些相似的向量存储多次，我们可以只进行一次存储，然后创建指向它们的指针。此时我们可以将空间的成本降低到$O(N\log_2K+KDC)$,其中$O(N\log_2K)$项表示$N$个向量需要指定它属于$K$个簇中的哪一个（指针）;$O(KDC)$表示我们存储码本中所有的$K$个向量所需要的空间大小，其中每个向量的维度为$D$，一般情况下，第一项在内存消耗上是占据主导的，所以编码方案的速率(rate)(每个目标所需要的位数)为$O(\log_2K)$，较$O(DC)$而言要小得多。

矢量量化的一个应用是图像压缩。考虑一个大小为$N=200 \times 320 = 64,000$的图片，如图11.12所示，这是一张灰度图，所以$D=1$。如果我们使用一个字节存储一个像素(灰度图的像素值范围为0-255)，则$C=8$，所以我们需要$NC=512,000$位的空间表示这张图片。为了压缩图片，我们需要$N\log_2K+KC$位。当$K=4$时，需要的空间大约为$128\rm{kb}$，压缩因子为4。当$K=4$时，空间为$192\rm{kb}$，压缩因子为2.6，其**感知损失**(perceptual loss)几乎可以忽略不计(见图11.12(b))。如果我们对像素之间的相关性进行建模，我们可以实现更大的压缩，举例来说，如果我们对$5\times5$的区域进行编码(就像JPEG所使用的方式)。这是因为残差(与模型预测值之间的区别)将会更小，从而使用更少的位数进行编码(**最后一句话待理解**)。

#### 11.4.2.7 初始化和避免局部最小值

K-means和EM算法都需要初始化。通常情况下，是随机选择K个数据点作为初始的簇的中心。或者我们可以采用序列的方式挑选中心点，从而尽可能地“覆盖”整个数据集。也就是说，我们均匀地随机采样一个初始的点。然后从剩余点中选取每个后续点，其概率与其到点最近的聚类中心的平方距离成比例。这被称为**最远距离点聚簇(farthest point clustering)**或者**k-means++**。令人惊讶的是，这个简单的技巧可以证明，失真绝不会比$O(\log K)$更糟糕。

语音识别社区中常用的启发式方法是逐渐“增长”GMM：我们最初根据混合权重给每个聚簇一个分数; 在每轮训练之后，我们考虑将具有最高分数的聚簇分成两个，其中新的质心是原始质心的随机扰动，并且新分数是旧分数的一半。 如果新聚簇的分数太低或方差太小，则会将其删除。 我们以这种方式继续，直到达到所需的簇数。 参见（Figueiredo 和 Jain 2002）的类似增量方法。

#### 11.4.2.8 最大后验估计

通常情况下，MLE可能会导致过拟合。而在GMMs问题中，过拟合问题尤其严重。为了理解这个问题，假设$\Sigma_k=\sigma_k^2I$，且$K=2$。如果我们将其中一个聚簇的中心点，比如说$\mathbf{\mu}_2$设置为一个单独的数据点$\mathbf{x}_1$，那么我们有可能获得一个无穷大的似然值。因为在这种情况下，其似然函数为:
$$
\mathcal{N}(\mathbf{x}_1|\mathbf{\mu}_2,\sigma_2^2I)=\frac{1}{\sqrt{2\pi\sigma_2^2}}e^0 \tag{11.39}
$$
所以当$\sigma_2\rightarrow0$时，上式将趋近于无穷大，如图11.13(a)所示。我们将上述现象称之为"(崩溃方差问题)collapsing variance problem"。

对此的简单解决方案是使用MAP估计。 新的辅助函数是完全数据对数似然的期望值加上对数先验：

$$
Q^\prime(\mathbf{\theta},\mathbf{\theta}^{old})=\left[\sum_i\sum_kr_{ik}\log\pi_{ik}+\sum_i\sum_kr_{ik}\log p(\mathbf{x}_i|\mathbf{\theta}_k)\right]+\log p(\mathbf{\pi})+\sum_k\log p(\mathbf{\theta}_k) \tag{11.40}
$$
值得注意的是步骤E并没有发生变化，但是步骤M需要进行调整，这一点我们会在下文解释。

对于混合权重的先验分布而言，很自然地是使用一个狄里克莱先验，$\mathbf{\pi} \sim \rm{Dir}(\mathbf{\alpha})$,因为该分布与类别分布是共轭分布。最大后验估计为：
$$
\pi_k = \frac{r_k+\alpha_k-1}{N+\sum_k\alpha_k-K} \tag{11.41}
$$
如果我们使用一个均匀先验分布，则令$\alpha_k=1$，这将使上式退化为公式11.28。

类条件分布的参数的先验分布$p(\mathbf{\theta}_k)$取决于条件分布的形式。我们接下来讨论GMM的案例。

为了简单起见，让我们考虑一个如下的共轭先验形式：
$$
p(\mathbf{\mu}_k,\mathbf{\Sigma}_k)={\rm{NIW}}(\mathbf{\mu}_k,\mathbf{\Sigma}_k|\mathbf{m}_0,\kappa_0,\nu_0,\mathbf{S}_0)\tag{11.42}
$$
根据章节4.6.3的内容，MAP估计由下式给定：
$$

$$
我们现在说明在GMM的背景下使用MAP估计替代ML估计的好处。 我们使用ML或MAP估计将EM算法应用于D维空间中的一些合成数据。 如果存在涉及奇异矩阵的数值问题，我们将该试验视为“失败”。 对于每个维度，我们进行5次随机试验。 结果如图11.13（b）所示，使用N = 100。我们看到，一旦维度D变大，ML估计就会崩溃，而MAP估计从未遇到过类似的数值问题。

当使用MAP估计时，我们需要指定超参数。此处我们介绍一些简单的设置这些参数的启发方式。我们可以令$\kappa_0=0$，从而使得$\mathbf{\mu}_k$不被正则化，因为数值问题只是因为$\mathbf{\Sigma}_k$而引起的。在这种情况下，MAP估计简化为$\hat{\mathbf{\mu}}_k=\bar{\mathbf{x}}_k$，$\hat{\mathbf{\Sigma}}_k=\frac{\mathbf{S}_0+\mathbf{S}_k}{\nu_0+r_k+D+2}$，这样至少在形式上就不会那么吓人了。

接下来，我们讨论如何设置$\mathbf{S}_0$。一种可能是使用
$$
\mathbf{S}_0=\frac{1}{K^{1/D}}{\rm{diag}}(s_1^2,...,s_D^2)\tag{11.48}
$$
其中$s_j=(1/N)\sum_{i=1}^{N}(x_{ij}-\bar{x}_j)^2$为维度$j$的共享方差。(之所以存在$\frac{1}{K^{1/D}}$这一项是因为每个椭球体最终的体积为$|\mathbf{S}_0|=\frac{1}{K}|{\rm{diag}}(s_1^2,...,s_D^2)|$.)参数$\nu_0$控制着我们对这个先验分布的置信程度。我们可以使用的最弱的先验分布为$\nu_0=D+2$,当然这个先验依然是合适的，所以通常情况下我们选择这个参数设置。

### 11.4.3 EM算法用于混合专家模型

我们可以直接使用EM算法拟合混合专家模型。完整数据的对数似然的期望值由下式给定：
$$

$$
所以步骤E与标准的混合模型并无两样。除了在计算$r_{ik}$时需要将$\pi_k$替换为$\pi_{i,k}$。

在步骤M，我们需要关于$\mathbf{w}_k$，$\sigma_k^2$和$\mathbf{V}$来最大化$Q(\mathbf{\theta},\mathbf{\theta}^{old})$。对于模型k的回归参数而言，目标函数的形式如下:
$$
Q(\mathbf{\theta}_k,\mathbf{\theta}^{old})=\sum_{i=1}^Nr_{ik}\left\{-\frac{1}{\sigma_k^2}(y_i-\mathbf{w}_k^T\mathbf{x}_i)\right\} \tag{11.52}
$$
上式可以被称为最小二乘问题的加权版本，这与直觉上是一致的：如果$r_{ik}$很小，那么在估计模型$k$的参数时，数据$i$将不再那么重要。根据8.3.4节，我们可以很快写出MLE为：
$$
\mathbf{w}_k=(\mathbf{X}^T\mathbf{R}_k\mathbf{X})^{-1}\mathbf{X}^T\mathbf{R}_k\mathbf{y} \tag{11.53}
$$
其中$\mathbf{R}_k={\rm{diag}}(r_{:,k})$。方差的MLE为:
$$
\sigma_k^2=\frac{\sum_{i=1}^Nr_{ik}(y_i-\mathbf{w}_k^T\mathbf{x}_i)^2}{\sum_{i=1}^Nr_{ik}} \tag{11.54}
$$
我们用门控参数$\mathbf{V}$的估计值代替无条件混合权重$\mathbf{\pi}$的估计值。目标函数具有以下形式：
$$
l(\mathbf{V})=\sum_i\sum_kr_{ik}\log\pi_{i,k} \tag{11.55}
$$
我们认为这等效于方程式8.34中多项式逻辑回归的对数似然性，只是我们用“软”的K编码$\mathbf{r}_i$替换了“硬”的C编码的$\mathbf{y}_i$。 因此，我们可以通过将逻辑回归模型拟合到软目标标签来估计参数$\mathbf{V}$。

### 11.4.4 EM算法用于含隐变量的DGMs

我们可以将用于混合专家模型的EM算法背后的思想泛化到计算任意DGM的MLE或者MAP估计。我们可以使用基于梯度的优化方法，但使用EM算法更加简单：在步骤E，我们只需要估计隐变量，但是在步骤M，我们将使用完整的数据计算MLE。我们在下文给出细节。

为便于表达，我们将假设所有的CPDs为表格类型的分布。根据章节10.4.2，我们将每个CPT写成如下形式：
$$
p(x_{it}|\mathbf{x}_{i,pa(t)},\mathbf{\theta}_t)=\prod_{c=1}^{K_{pa(t)}}\prod_{k=1}^{K_t}\theta_{tck}^{\mathbb{I}(x_{it}=i,\mathbf{x}_{i, pa(t)}=c)} \tag{11.56}
$$
完整数据的对数似然如下式：
$$
\log p(\mathcal{D}|\mathbf{\theta})=\sum_{t=1}^{V}\sum_{c=1}^{K_{pa(t)}}\sum_{k=1}^{K_t}N_{tck}\log\theta_{tck} \tag{11.57}
$$
其中$N_{tck}=\sum_{i=1}^{N}\mathbb{I}(x_{it}=i,\mathbf{x}_{i,pa(t)}=c)$为经验计数。所以完整数据的对数似然的期望值为:
$$
\mathbb{E}[\log p(\mathcal{D}|\mathbf{\theta})]=\sum_t\sum_c\sum_k\bar{N}_{tck}\log\theta_{tck} \tag{11.58}
$$
其中
$$
\bar{N}_{tck}=\sum_{i=1}^{N}\mathbb{E}[\mathbb{I}(x_{it}=i,\mathbf{x}_{i,pa(t)}=c)]=\sum_ip(x_{it}=k,\mathbf{x}_{i,pa(t)}=c|\mathcal{D}_i)\tag{11.59}
$$
其中$\mathcal{D}_i$为样本$i$中的所有可见变量。

上式中$ p(x_{it},\mathbf{x}_{i,pa(t)}|\mathcal{D}_i,\mathbf{\theta}) $为**族边缘(family marginal)**，可以使用GM推理算法获得。$\bar{N}_{tjk}$为期望充分统计量，也就是E步骤的输出。

基于这些EES，M步骤具备如下的简单形式：
$$
\hat{\theta}_{tck}=\frac{\bar{N}_{tck}}{\sum_{k^\prime}\bar{N}_{tjk^\prime}} \tag{11.60}
$$
这可以通过以下方法证明：将Lagrange乘子（强制执行约束$\sum_k\theta_{tjk}=1$）添加到预期的完整数据对数似然率，然后分别优化每个参数矢量$\mathbf{\theta}_{tc}$。 我们可以修改它以通过Dirichlet事先执行MAP估计，方法很简单，将伪计数添加到预期计数。

### 11.4.5 学生分布的EM算法

### 11.4.6 probit回归的EM算法

### 11.4.7 EM的理论基础

本节，我们将展示EM算法是如何单调的增加观测数据的对数似然，直到它到达一个局部最大值的（或者一个鞍点，尽管这样的点通常是不稳定的）。我们的推导同样适用于我们接下来讨论的EM算法的各种变体。

#### 11.4.7.1 完整数据的对数似然的期望值是一个下确界

考虑任意一个关于隐变量的分布$q(\mathbf{z}_i)$。观测值的对数似然可以写成如下形式：
$$
l(\mathbf{\theta}) \triangleq \sum_{i=1}^N \log\left[\sum_{\mathbf{z}_i}p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})\right]=\sum_{i=1}^N \log\left[\sum_{\mathbf{z}_i}q(\mathbf{z}_i)\frac{p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})}{q(\mathbf{z}_i)}\right] \tag{11.84}
$$
其中$\log(u)$是一个凹函数，所以根据琴森不等式(公式2.113)，我们有如下的下确界：
$$
l(\mathbf{\theta})\ge \sum_i\sum_{\mathbf{z}_i}q_i(\mathbf{z}_i)\log\frac{p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})}{q_i(\mathbf{z}_i)} \tag{11.85}
$$
我们将下确界写成如下的形式：
$$
Q(\mathbf{\theta},q) \triangleq \sum_i\mathbb{E}_{q_i}[\log p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})] + \mathbb{H}(q_i) \tag{11.86}
$$
其中$\mathbb{H}(q_i)$为$q_i$的熵。

上述结论对于所有大于0的分布$q$都是成立的。我们应该选择哪一个呢？直觉上，我们应该选择一个分布$q$，使得等号刚好成立。第$i$个样本的下确界为：
$$
\begin{align}
L(\mathbf{\theta},q_i) & = \sum_{\mathbf{z}_i}q_i(\mathbf{z}_i)\log\frac{p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})}{q_i(\mathbf{z}_i)}  \tag{11.87} \\
& = \sum_{\mathbf{z}_i}q_i(\mathbf{z}_i)\log\frac{p(\mathbf{z}_i|\mathbf{x}_i,\mathbf{\theta})p(\mathbf{x}_i|\mathbf{\theta})}{q_i(\mathbf{z_i})} \tag{11.88} \\
& = \sum_{\mathbf{z}_i}q_i(\mathbf{z}_i)\log\frac{p(\mathbf{z}_i|\mathbf{x}_i,\mathbf{\theta})}{q_i(\mathbf{z}_i)} + \sum_{\mathbf{z}_i}q_i(\mathbf{z}_i)\log p(\mathbf{x}_i|\mathbf{\theta}) \tag{11.98} \\
& = -\mathbb{KL}(q_i(\mathbf{z}_i)||p(\mathbf{z}_i|\mathbf{x}_i,\mathbf{\theta})) + \log p(\mathbf{x}_i|\mathbf{\theta}) \tag{11.90}
\end{align} 
$$
其中$p(\mathbf{x}_i|\mathbf{\theta})$与$q_i$无关，所以如果我们令$q_i(\mathbf{z}_i)=p(\mathbf{z}_i|\mathbf{x}_i,\mathbf{\theta})$，就可以最大化上式。当然，因为$\mathbf{\theta}$是未知的，所以取而代之的，我们令$q_i^t(\mathbf{z}_i)=p(\mathbf{z}_i|\mathbf{x}_i,\mathbf{\theta}^t)$，其中$\mathbf{\theta}^t$为我们在步骤$t$时的估计值，也就是步骤E的输出。

在已知$q^t$的情况下，我们将其带入到下确界中:
$$
Q(\mathbf{\theta},q^t)=\sum_i\mathbb{E}_{q_i^t}[\log p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})] + \mathbb{H}(q_i^t) \tag{11.91}
$$
我们将第一项作为完整数据的对数似然的期望值。第二项与$\mathbf{\theta}$无关。所以步骤M通常就变为：
$$
\mathbf{\theta}^{t+1} = \mathop{\arg\max}_\mathbf{\theta}Q(\mathbf{\theta},\mathbf{\theta}^t)=\mathop{\arg\max}_\mathbf{\theta} \sum_i\mathbb{E}_{q_i^t}[\log p(\mathbf{x}_i,\mathbf{z}_i|\mathbf{\theta})] \tag{11.92}
$$
接下来就是比较巧妙的地方了。既然我们令$q_i^t(\mathbf{z}_i)=p(\mathbf{z}_i|\mathbf{x}_i,\mathbf{\theta}^t)$，那么$KL$散度也就为0，所以$L(\mathbf{\theta}^t,q_i)=\log p(\mathbf{x}_i|\mathbf{\theta}^t)$，因此:
$$
Q(\mathbf{\theta}^t,\mathbf{\theta}^t)=\sum_i\log p(\mathbf{x}_i|\mathbf{\theta}^t)=l(\mathbf{\theta}^t) \tag{11.93}
$$
我们发现在步骤E之后，这个下确界是紧凑的。既然这个下确界"触碰到了"目标函数，那么最大化这个下确界将会“抬高”这个函数本身。也就是说，步骤M肯定可以通过调整参数实现观测数据的似然值增加（除非它已经到达了一个局部最大值）。

图11.16说明了这个过程。其中红色的虚线是实际的函数（观测数据的对数似然）。蓝色实线是下确界，对应$\mathbf{\theta}^t$；该函数在$\mathbf{\theta}^t$处与目标函数相接触。然后我们通过调整$\mathbf{\theta}^{t+1}$来最大化下确定(蓝色曲线)，并在这一点拟合一个新的下确界(绿色虚线)。这个新的下确界的最大值在$\mathbf{\theta}^{t+2}$处，以此类推。（将该方法与图8.4(a)中的牛顿法相比较，在该方法中，优化与二次函数近似重复地进行）。

#### 11.4.7.2 EM算法单调地增加观测数据的对数似然

我们现在证明EM算法可以单调的增加观测数据的对数似然，直到到达一个局部最优解。我们有：
$$
l(\mathbf{\theta}^{t+1})\ge Q(\mathbf{\theta}^{t+1},\mathbf{\theta}^t)\ge Q(\mathbf{\theta}^t,\mathbf{\theta}^t) = l(\mathbf{\theta}^t) \tag{11.94}
$$
其中第一个不等式成立的原因在于$Q(\mathbf{\theta},.)$是$l(\mathbf{\theta})$的下界；第二个不等式成立的原因在于，根据定义$Q(\mathbf{\theta}^{t+1},\mathbf{\theta}^t)=\max_\mathbf{\theta}Q(\mathbf{\theta},\mathbf{\theta}^t)\ge Q(\mathbf{\theta}^t,\mathbf{\theta}^t)$。最后一个等式成立的原因可参照式11.93.

根据上述结果，如果你不能观察到观测数据的对数似然渐进地增加，那么在你的数学上或者代码上存在问题。（如果你使用的是最大后验估计，你必须在目标函数上加上对数先验项。）这是一个强有力的调试工具。

### 11.4.8 在线EM算法

### 11.4.9 其他EM算法变体

## 11.5 潜变量模型的模型选择

当使用LVMs时，我们必须指定潜变量的数量，它控制着模型的复杂度。特别地，在混合模型中，我们必须指定参数$K$,也就是聚簇的数量。对这些参数的选择属于模型选择任务的一种。我们在下面讨论一些方法。

### 11.5.1 概率模型的模型选择

正如5.3节所介绍的，最优的贝叶斯方法可以用来挑选具备最大边缘似然的模型，即$K^*=\arg \max_kp(\mathcal{D}|K).$

上述方法存在两个问题。首先，估计LVMs的边缘似然是一件相当困难的事情。在实际过程中，一些简单的近似方法，比如BIC，可以使用。除此之外，我们可以使用交叉验证的似然值作为一个性能指标，尽管这样可能会比较慢，因为它需要训练每个模型$F$次，其中$F$为分包的数量。

第二个问题在于需要在大量的模型中进行检索。常规的方法是采用穷举法，搜索所有候选的$K$值。然而，有些时候我们可以将模型设置为它的最大尺寸，然后依靠贝叶斯奥卡姆剃刀的方法“剔除”掉一些不想要的成分。其中一个例子可参照$21.6.1.6$节，在那里我们讨论了变分贝叶斯。

另一种方法是在模型空间中执行随机抽样。 传统的方法，例如（Green 1998，2003； Lunn等，2009），是基于可逆跳MCMC的，并使用出生移动来提议新的中心，而死亡移动来杀死旧的中心。 但是，这可能很慢并且难以实现。 一种更简单的方法是使用Dirichlet过程混合模型，该模型可以使用Gibbs采样进行拟合，但仍然允许无限制数量的混合成分； 有关详细信息，请参见第25.2节。

也许令人惊讶的是，这些基于采样的方法可能比分别评估每个K的质量的简单方法要快。 原因是为每个K拟合模型通常很慢。 相比之下，采样方法通常可以快速确定K的某个值很差，因此不需要在后验的那部分浪费时间。

### 11.5.2 非概率模型的模型选择

如果我们不使用概率模型怎么办？ 例如，如何为$K-means$算法选择$K$？ 由于这不对应于概率模型，因此不存在“可能性”，因此上述方法均无法使用。

“可能性”的一个明显替代是重构误差。 使用模型复杂度$K$定义数据集$\mathcal{D}$的平方重建误差，如下所示:
$$
E(\mathcal{D},K)=\frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}}||\mathbf{x}_i-\hat{\mathbf{x}}_i||^2 \tag{11.95}
$$
在$K-means$中，重构方式为$\hat{\mathbf{x}}_i=\mathbf{\mu}_{z_i}$,其中$z_i=\arg\min_k||\mathbf{x}_i-\mathbf{\mu}_k||_2^2$，这一点我们在11.4.2.6节中有过解释。

图11.20(a)绘制了$K-means$方法中在**测试集**上的重构误差。我们发现随着模型复杂度的增加，重构误差也会下降！之所以会出现这种现象，是因为：当我们在$K-means$方法中增加越来越多的质心时，我们可以将质心紧密地“覆盖”整个数据空间，如图11.21(b)所示。因此，随着K的增加，任何给定的测试向量都更有可能找到一个接近的原型来精确地表示它，从而减少重构误差。 但是，如果使用概率模型（例如GMM）并绘制负对数似然，则会在测试集上获得常见的U形曲线，如图11.20（b）所示。

在监督学习中，我们总是可以使用交叉验证在不同复杂度的非概率模型之间进行选择，但是在无监督学习中则不是这种情况。 尽管这不是一个新颖的发现（例如，在本领域的标准参考文献之一（Hastie et al.2009，p519）中被提及），但它可能并未得到应有的广泛认可。 实际上，它是支持概率模型的更引人注目的论据之一。

鉴于交叉验证不起作用，并且假设一个人不愿意使用概率模型（出于某些奇怪的原因...），那么如何选择$K$？ 最常见的方法是在训练集上绘制相对于$K$的重构误差，并尝试识别曲线中的转折点。 原因在于，对于$K <K^*$，其中$K ^*$是集群的“真实”数目，由于我们将不应该分在一块的东西划分到了一块，因此误差函数会下降的很快。 但是，对于$K> K^*$，我们将那些“自然”存在簇也分开了，所以其误差函数下降得并不会太快。（**该段话难以理解**）

可以通过使用缺口统计(**gap statistic**)来自动找到这种转折点（Tibshirani等，2001）。 然而，如图11.20（a）所示，识别这种转折点可能很困难，因为损失函数通常会逐渐下降。 第12.3.2.1节介绍了另一种“发现转折点”的方法。

## 11.6 含缺失数据的模型拟合

假设我们希望通过最大似然拟合联合密度模型，但是由于缺少数据（通常由NaN表示），我们的数据矩阵中存在“空洞”。 更正式地说，如果观察到数据情况i的分量$j$，则让$O_{ij} = 1$，否则让$O_{ij} = 0$。 令$\mathbf{X}_v = \{x_{ij}：O_{ij} = 1\}$是可见数据，而$\mathbf{X}_h = \{x_{ij}：O_{ij} = 0\}$是丢失或隐藏的数据。 我们的目标是计算:
$$
\hat{\mathbf{\theta}}=\arg \min \limits_\mathbf{\theta}p(\mathbf{X}_v|\mathbf{\theta},\mathbf{O}) \tag{11.96}
$$
在随机缺失的假设下(见8.6.2节)，我们有:
$$
p(\mathbf{X}_v|\mathbf{\theta},\mathbf{O})=\prod_{i=1}^Np(\mathbf{x}_{iv}|\mathbf{\theta})
$$
其中$\mathbf{x}_{iv}$为第$i$行的向量，列的索引为集合$\{j:O_{ij}=1\}$.所以对数似然的形式为：
$$
\log p(\mathbf{X}_v|\mathbf{\theta})=\sum_i \log p(\mathbf{x}_{iv}|\mathbf{\theta}) \tag{11.98}
$$
其中：
$$
p(\mathbf{x}_{iv}|\mathbf{\theta})=\sum_{\mathbf{x}_{ih}}p(\mathbf{x}_{iv},\mathbf{x}_{ih}|\mathbf{\theta}) \tag{11.99}
$$
其中$\mathbf{x}_{ih}$为第$i$个样本的隐变量（为了符号上的表达简单，我们假设为离散变量）。将上式进行综合：
$$
\log p(\mathbf{X}_v|\mathbf{\theta})=\sum_i \log\left[\sum_{\mathbf{x}_{ih}}p(\mathbf{x}_{iv},\mathbf{x}_{ih}|\mathbf{\theta})\right] \tag{11.100}
$$
不幸的是，这个目标很难实现。 因为我们无法将对数推入总和内。 但是，我们可以使用EM算法来计算局部最优值。 我们在下面给出一个例子。

### 11.6.1 EM算法用于含缺失数据的模型MLE







