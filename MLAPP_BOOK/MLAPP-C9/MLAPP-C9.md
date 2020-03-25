

# 9.广义线性模型和指数族分布

## 9.1 介绍
&emsp;&emsp;我们已经遇到过很多种概率分布：高斯分布，伯努利分布，学生t分布，均匀分布，伽玛分布等等。结果表明这些分布中大部分分布都属于一个被称为**指数族**(exponential family)的分布。本章，我们将讨论这个分布族的各种性质。这将允许我们推导出应用更加广泛的理论和算法。
&emsp;&emsp;我们将看到我们如何轻易地使用指数族中的任意成员作为类条件概率密度，从而实现一个生成式分类器。除此之外，我们将讨论，当目标变量服从指数族分布时，如何构造一个判别式模型，其中该分布的期望是输入变量的线性函数；这被称为广义线性模型，它是将逻辑回归泛化到其他类型的目标变量。

## 9.2 指数族分布
在介绍指数族之前，我们需要指出几点为什么它很重要的理由：  

- 结果表明，在某些正则条件下，指数族是具有有限大小充分统计量的唯一的分布族，意味着我们可以将数据压缩到固定大小的规模而不需要损失信息。这对于在线学习特别的重要，这一点我们会在后面看到。
- 指数族是唯一存在共轭先验的分布族，从而简化了后验分布的计算(见9.2.5节)。
- 在具备某些人为约束的情况下，指数族分布是需要最少假设的分布族(见9.2.6节)。
- 指数族分布是广义线性模型的核心，这一点将在9.3节讨论。
- 指数族分布是变分推断的核心，这一点将在21.2节进行讨论。
### 9.2.1 定义
一个概率密度函数或者概率质量分布$p(\mathbf{x}|\mathbf{\theta})​$其中$\mathbf{x}=(x_1,...,x_m)\in\mathcal{X}^m​$，$\mathbf{\theta}\in\Theta\subseteq\mathbb{R}^d​$，如果具备如下的形式，则属于**指数族(exponential family)**:
$$
\begin{align}
p(\mathbf{x}|\mathbf{\theta}) & =\frac{1}{Z(\mathbf{\theta})}h(\mathbf{x})exp[\mathbf{\theta}^T\phi(\mathbf{x})] \tag{9.1}  \\
& = h(\mathbf{x})exp[\mathbf{\theta}^T\phi(\mathbf{x})-A(\mathbf{\theta})] \tag{9.2} \\
\end{align}
$$
其中
$$
\begin{eqnarray*}
Z(\mathbf{\theta}) & = & \int_{\mathcal{X}^m}h(\mathbf{x})exp[\mathbf{\theta}^T\phi(\mathbf{x})]d\mathbf{x} \tag{9.3} \\
A(\mathbf{\theta}) & = & \log Z(\mathbf{\theta})  \tag{9.4}
\end{eqnarray*}
$$
此处$\mathbf{\theta}$被称为**自然参数（natural parameters）**或者**典范参数（canonical parameters）**，$\mathbf{\phi}(\mathbf{x})\in\mathbb{R}^d$被称为一个**充分统计量（sufficient statistics）**向量，$Z(\mathbf{\theta})$被称为**配分函数(partition function)**，$A(\mathbf{\theta})$被称为对数**配分函数(log partition function)**或者**累积函数(cumulant function)**，$h(\mathbf{x})$是一个尺度常数，通常情况下等于1.如果$\mathbf{\phi}(\mathbf{x})=\mathbf{x}$，我们称它是一个**自然指数族（natural exponential family）**。

&emsp;&emsp;公式9.2的一般化形式为：
$$
p(\mathbf{x}|\mathbf{\theta})=h(\mathbf{x})\exp[\eta(\mathbf{\theta})^T\mathbf{\phi(\mathbf{x})}-A(\eta(\mathbf{\theta}))] \tag{9.5}
$$
其中$\eta$为一个映射函数，它将参数$\mathbf{\theta}$映射到典范参数$\mathbf{\eta}=\eta(\mathbf{\theta})$。如果$dim(\mathbf{\theta})<dim(\eta(\mathbf{\theta}))$，那么它被称为**曲指数族（curved exponential family）**，意味着我们的充分统计量大于参数数量。如果$\eta(\mathbf{\theta})=\mathbf{\theta}$，那么模型被称为**典范形式（canonical form）**。在后面的内容中我们将假设使用模型的典范形式，除非我们作特殊的说明。

### 9.2.2 例子

现在让我们考虑一些例子，从而让前文的内容更加清晰。

#### 9.2.2.1 伯努利分布

对于变量$x\in\left\{0,1\right\}$，其伯努利分布可以写成如下的指数族形式：
$$
Ber(x|\mu)=\mu^x(1-\mu)^{1-x}=\exp[x\log(\mu)+(1-x)\log(1-\mu)]=\exp[\mathbf{\phi}(x)^T\mathbf{\theta}] \tag{9.6}
$$
其中$\mathbf{\phi}(x)=[\mathbb{I}(x=0),\mathbb{I}(x=1)]$，$\mathbf{\theta}=[\log(\mu),\log(1-\mu)]$。然而，这种表示形式是**过完备（over-complete）**的，因为在特征之间存在线性相关性：
$$
\mathbf{1}^T\mathbf{\phi}(x)=\mathbb{I}(x=0)+\mathbb{I}(x=1)=1 \tag{9.7}
$$
因此我们没有办法对参数$\mathbf{\theta}$进行唯一的确定。通常情况下我们需要表示方法是**最小的(minimal)**，意味着与这个分布有关的参数$\mathbf{\theta}$是唯一的。在这种情况下，我们可以定义:
$$
Ber(x|\mu)=(1-\mu)\exp\left[x\log\left(\frac{\mu}{1-\mu}\right)\right] \tag{9.8}
$$
现在我们有$\phi(x)=x,\theta=\log\left(\frac{\mu}{1-\mu}\right)$,这便是对数几率，$Z=1/(1-\mu)$，我们可以使用期望参数回溯典范参数，其方法为：
$$
\mu=sigm(\theta)=\frac{1}{1+e^{-\theta}} \tag{9.9}
$$

#### 9.2.2.2 Multinoulli

我们可以将multinoulli表示成最小指数族分布，采用如下的形式（其中$x_k=\mathbb{I}(x=k)$）:
$$
\begin{align}
Cat(x|\mathbf{\mu}) & = \prod_{k=1}^K \mu_k^{x_k}\log\mu_k =\exp\left[\sum_{k=1}^K x_k\log\mu_k\right] \tag{9.10} \\
& = \exp\left[\sum_{k=1}^{K-1} x_k\log\mu_k + \left(1-\sum_{k=1}^{K-1}x_k\right)\log\left(1-\sum_{k=1}^{K-1}\mu_k\right)\right] \tag{9.11} \\
& = \exp\left[\sum_{k=1}^{K-1}x_k\log\left(\frac{\mu_k}{1-\sum_{j=1}^{K-1}\mu_j}\right)+\log\left(1-\sum_{k=1}^{K-1}\mu_k\right)\right] \tag{9.12} \\
& = \exp\left[\sum_{k=1}^{K-1}x_k\log\left(\frac{\mu_k}{\mu_K}\right)+\log\mu_K\right] \tag{9.13}
\end{align} 
$$
式中$\mu_K=1-\sum_{k=1}^{K-1}\mu_k$。我们可以将其写成指数族分布的形式：
$$
\begin{eqnarray*}
Cat(x|\mathbf{\theta}) & = & \exp(\mathbf{\theta}^T\mathbf{\phi}(\mathbf{x})-A(\mathbf{\theta})) \tag{9.14} \\
\mathbf{\theta} & = & \left[\log\left(\frac{\mu_1}{\mu_K}\right),...,\log\left(\frac{\mu_{K-1}}{\mu_K}\right)\right] \tag{9.15} \\
\mathbf{\phi}({x}) & = & [\mathbb{I}(x=1),...,\mathbb{I}(x=K-1)] \tag{9.16}
\end{eqnarray*}
$$
我们可以从典范参数中回溯期望参数，使用下式实现：
$$
\mu_k=\frac{e^{\theta_k}}{1+\sum_{j=1}^{K-1}e^{\theta_j}} \tag{9.17}
$$
进而，我们可以得到：
$$
\mu_K = 1-\frac{\sum_{j=1}^{K-1}e^{\theta_j}}{1+\sum_{j=1}^{K-1}e^{\theta_j}}=\frac{1}{1+\sum_{j=1}^{K-1}e^{\theta_j}} \tag{9.18}
$$
所以：
$$
A(\mathbf{\theta})=\log\left(1+\sum_{k=1}^{K-1}e^{\theta_k}\right) \tag{9.19}
$$
如果我们定义$\theta_K=0$，我们可以定义$\mathbf{\mu}=\mathcal{S}(\mathbf{\theta)},A(\mathbf{\theta})=\log\sum_{k=1}^K e^{\theta_k}$，其中$\mathcal{S}$为softmax函数（式4.39）。

#### 9.2.2.3 单变量高斯分布

单变量高斯分布可以写成如下的指数族分布形式：
$$
\begin{align}
\mathcal{N}(x|\mu,\sigma^2) & = \frac{1}{(2\pi\sigma^2)^{\frac{1}{2}}}\exp\left[-\frac{1}{2\sigma^2}(x-\mu)^2\right] \tag{9.20} \\
& = \frac{1}{(2\pi\sigma^2)^{\frac{1}{2}}}\exp\left[-\frac{1}{2\sigma^2}x^2+\frac{\mu}{\sigma^2}x-\frac{1}{2\sigma^2}\mu^2\right] \tag{9.21} \\
& = \frac{1}{Z(\mathbf{\theta})}\exp(\mathbf{\theta}^T\mathbf{\phi}(x)) \tag{9.22}
\end{align}
$$
其中
$$
\begin{eqnarray*}
\mathbf{\theta} &= &\dbinom{\mu/\sigma^2}{\frac{-1}{2\sigma^2}} \tag{9.23}\\
\mathbf{\phi} &=& \dbinom{x}{x^2} \tag{9.24} \\
Z(\mu, \sigma^2) &=& \sqrt{2\pi}\sigma\exp[\frac{\mu^2}{2\sigma^2}] \tag{9.25} \\
A(\mathbf{\theta}) & = & \frac{-\theta_1^2}{4\theta_2}-\frac{1}{2}\log(-2\theta_2)-\frac{1}{2}\log(2\pi) \tag{9.26}
\end{eqnarray*}
$$

#### 9.2.2.4 反例

不是所有的分布都属于指数族。举例来说，均匀分布$X$~$Unif(a,b)$就不属于指数族，因为分布的支撑集与参数直接相关。同样，学生T分布（章节11.4.5）也不属于，因为它不具备指数族分布的必要形式。

### 9.2.3 对数配分函数

指数族分布的一个重要性质是对数配分函数的导数可以用来生成充分统计量的**累积量**（原文住：概率分布的一阶和二阶累计量分别为期望$\mathbb{E}[X]$和方差$Var[X]$，一阶矩和二阶矩分别为$\mathbb{E}[X],\mathbb{E}[X^2]$。）基于这个原因，$A(\mathbf{\theta})$又被称为**累积函数(cumulant function)**。我们会以只有一个参数逇概率分布为例，来证明上述的相关结论，这个结论同样适用于K个参数的情况。对于一阶导数，我们有：
$$
\begin{align}
\frac{dA}{d\theta} & =\frac{d}{d\theta}\left(\log\int_{}\exp(\theta\phi(x))h(x)dx\right) \tag{9.27} \\
& = \frac{\frac{d}{d\theta}\int_{}\exp(\theta\phi(x))h(x)dx} {\int_{}\exp(\theta\phi(x))h(x)dx} \tag{9.28} \\
& = \frac{\int_{}\phi(x)\exp(\theta\phi(x))h(x)dx}{\exp(A(\theta))} \tag{9.29} \\
& = \int_{}\phi(x)\exp(\theta\phi(x)-A(\theta))h(x)dx \tag{9.30} \\
& = \int_{}\phi(x)p(x)dx = \mathbb{E}[\phi(x)] \tag{9.31}
\end{align}
$$
对于二阶导，我们有：
$$
\begin{align}
\frac{d^2A}{d\theta^2} & = \int_{}\phi(x)\exp(\theta\phi(x)-A(\theta))h(x)(\phi(x)-A^\prime(\theta))dx \tag{9.32} \\
& = \int_{}\phi(x)p(x)(\phi(x)-A^\prime(\theta))dx \tag{9.33} \\
& = \int_{}\phi^2(x)p(x)dx-A^\prime(\theta)\int_{}\phi(x)p(x)dx \tag{9.34} \\
& = \mathbb{E}[\phi^2(X)]-\mathbb{E}[\phi(x)]^2=var[\phi(x)] \tag{9.35}
\end{align}
$$
式中我们应用了之前的结论$A^\prime(\theta)=\frac{dA}{d\theta}=\mathbb{E}[\phi(x)]$.

在多变量情况，我们有：
$$
\frac{\partial^2A}{\partial\theta_i\partial\theta_j}=\mathbb{E}[\phi_i(x)\phi_j(x)]-\mathbb{E}[\phi_i(x)]\mathbb{E}[\phi_j(x)] \tag{9.36}
$$
所以
$$
\nabla^2A(\mathbf{\theta})=cov[\mathbf{\phi}(\mathbf{x})] \tag{9.37}
$$
因为协方差矩阵是正定矩阵，所以$A(\mathbf{\theta})$是一个凸函数（见章节7.3.3）。

#### 9.2.3.1 例子

举例来说，考虑伯努利分布的情况。我们有$A(\theta)=\log(1+e^\theta)$ ，所以分布的期望为:
$$
\frac{dA}{d\theta}=\frac{e^\theta}{1+e^\theta}=\frac{1}{1+e^{-\theta}}=sigm(\theta)=\mu \tag{9.38}
$$
方差为：
$$
\begin{align}
\frac{d^2A}{d\theta^2} & = \frac{d}{d\theta}(1+e^{-\theta})^{-1}=(1+e^{-\theta})^{-2}e^{-\theta} \tag{9.39} \\
& = \frac{e^{-\theta}}{1+e^{-\theta}}\frac{1}{1+e^{-\theta}}=\frac{1}{1+e^\theta}\frac{1}{1+e^{-\theta}}=(1-\mu)\mu \tag{9.40}
\end{align}
$$

### 9.2.4 指数族分布的MLE

指数族分布的似然函数具备如下的形式
$$
p(\mathcal{D}|\mathbf{\theta})=\left[\prod_{i=1}^N h(\mathbf{x}_i)\right]g(\mathbf{\theta})^N\exp\left(\mathbf{\eta}(\mathbf{\theta})^T[\sum_{i=1}^N\mathbf{\phi}(\mathbf{x}_i)]\right) \tag{9.41}
$$
我们发现充分统计量为N和
$$
\mathbf{\phi}(\mathcal{D})=[\sum_{i=1}^N\phi_1(\mathbf{x}_i),...,\sum_{i=1}^N\phi_K(\mathbf{x}_i)] \tag{9.42}
$$
举例来说，对于伯努利模型，我们有$\mathbf{\phi}=[\sum_{i}\mathbb{I}(x_i=1)]$，对于单变量高斯分布，我们有$\phi=[\sum_{i}x_i,\sum_{i}x_i^2]$.（我们同时需要样本的大小N）。

**Pitman-Koopman-Darmois**理论表明，在某些正则条件下，指数族分布是唯一一个具备有限充分统计量的分布。（此处，有限是指充分统计量的数量与数据集的大小数量无关。）

在这个理论中，有一个必要条件，分布的支撑集与参数无关。举一个这种类型的分布，考虑均匀分布：
$$
p(x|\theta)=U(x|\theta)=\frac{1}{\theta}\mathbb{I}(0\le x \le \theta)\tag{9.43}
$$
似然函数由下式给定：
$$
p(\mathcal{D}|\mathbf{\theta})=\theta^{-N}\mathbb{I}(0 \le \max\{x_i\} \le \theta) \tag{9.44}
$$
所以统计量为$N$和$s(\mathcal{D})=\max_ix_i$，其数量是有限的，但均匀分布不属于指数族分布，因为它的支撑集$\mathcal{X}$与参数相关。

我们现在讨论如何计算典范指数族分布的MLE。给定$N$个独立同分布的数据$\mathcal{D}=(x_1,...,x_N)$，其对数似然函数为：
$$
\log p(\mathcal{D}|\mathbf{\theta})=\mathbf{\theta}^T\mathbf{\phi}(\mathcal{D})-NA(\mathbf{\theta}) \tag{9.45}
$$
考虑到$-A(\mathbf{\theta})$是一个关于$\mathbf{\theta}$的凹函数，$\mathbf{\theta}^T\mathbf{\phi}(\mathcal{D})$是关于$\mathbf{\theta}$的线性函数。所以对数似然函数是一个凹函数，进而拥有一个唯一的全局最大值。为了推导出这个最大值，我们利用前文中推导出来的结论：对数配分函数的导数等于充分统计量的期望（见9.2.3）：
$$
\nabla_\mathbf{\theta}\log p(\mathcal{D}|\mathbf{\theta})=\mathbf{\phi}(\mathcal{D})-N\mathbb{E}[\mathbf{\phi}(\mathbf{X})] \tag{9.46}
$$
设置导数为0，我们发现在MLE解处，充分统计量的经验平均值必须等于模型理论的充分统计量的期望值，比如说$\hat{\mathbf{\theta}}$必须满足：
$$
\mathbb{E}[\mathbf{\phi}(\mathbf{X})]=\frac{1}{N}\sum_{i}^N\mathbf{\phi}(\mathbf{x}_i) \tag{9.47}
$$
这被称为**矩匹配(moment matching)**。举例来说，在伯努利分布中，我们有$\phi(X)=\mathbb{I}(X=1)$，所以$MLE$满足:
$$
\mathbb{E}[\phi(X)]=p(X=1)=\hat{\mu}=\frac{1}{N}\sum_{i}^N\mathbb{I}(x_i=1)\tag{9.48}
$$

### 9.2.5 指数族分布的贝叶斯方法*

### 9.2.6 指数族分布的最大熵推导*

尽管指数族分布在使用上十分方便，那它被广泛使用，有没有其他更深层次的原因呢？结果表明：在满足一些人为规定的约束前提下，指数族分布对数据所作出的额外假设是最少的。关于这一点我们将在下文介绍。特别地，假设我们已知的全部条件是某个特征或者函数的期望值：
$$
\sum_{\mathbf{x}}f_k(\mathbf{x})p(\mathbf{x})=F_k \tag{9.71}
$$
其中$F_k$是已知的约束，$f_k(\mathbf{x})$是一个任意的函数。**最大熵(maximum entropy, maxent)**原则是指：假设分布需要满足一些约束，在这些约束中，特定函数的真实矩与其经验矩需要匹配，我们需要选择熵最大的分布（接近均匀分布）。

为了在式9.71约束下使分布的熵最大化，同时需要满足约束$p(\mathbf{x})\ge 0,\sum_{\mathbf{x}}p(\mathbf{x})=1$，我们需要使用拉格朗日乘子法。其拉格朗日公式为：
$$
J(p,\mathbf{\lambda})=-\sum_{\mathbf{x}}p(\mathbf{x})\log p(\mathbf{x})+\lambda_0(1-\sum_{\mathbf{x}}p(\mathbf{x}))+\sum_{k}\lambda_k(F_k-\sum_{\mathbf{x}}p(\mathbf{x})f_k(\mathbf{x})) \tag{9.72}
$$
我们可以对上式关于分布$p$进行求导，但我们将使用一个更加简单的方法，在这种方法中，我们将$p$看作一个固定长度的向量（因为我们假设$\mathbf{x}$是离散的）。然后，我们有：
$$
\frac{\partial{J}}{\partial{p(\mathbf{x})}}=-1-\log p(\mathbf{x})-\lambda_0-\sum_{k}\lambda_kf_k(\mathbf{x}) \tag{9.73}
$$
令$\frac{\partial{J}}{\partial{p(\mathbf{x})}}=0$，得到：
$$
p(\mathbf{x})=\frac{1}{Z}\exp(-\sum_{k}\lambda_kf_k(\mathbf{x})) \tag{9.74}
$$
其中$Z=e^{1+\lambda_0}$。使用和为1的约束条件，我们有：
$$
1=\sum_{\mathbf{x}}p(\mathbf{x})=\frac{1}{Z}\sum_{\mathbf{x}}\exp(-\sum_{k}\lambda_kf_k(\mathbf{x})) \tag{9.75}
$$
所以归一化常数为：
$$
Z=\sum_{\mathbf{x}}\exp(-\sum_{k}\lambda_kf_k(\mathbf{x})) \tag{9.76}
$$
所以最大熵分布$p(\mathbf{x})$具备指数族分布的形式（9.2节），它又被称为**吉布斯分布（Gibbs distribution)**.

## 9.3 广义线性模型（GLMs）

线性回归和逻辑回归都属于**广义线性模型（generalized linear models,GLMs)**。在这些模型中，输出变量都服从指数族分布（9.2节），且分布的期望是输入的线性组合，有些情况下还会使用一个非线性函数，比如逻辑斯特函数。我们在下面讨论更多关于GLMs的细节。为了符号上的简便，我们将集中讨论标量情况。（这样我们就将multinomial 逻辑回归排除在外，但这只是为了简化表述。）

### 9.3.1 基础

为了理解GLMs，让我们首先考虑一个标量响应在非条件分布下的形式：
$$
p(y_i|\theta,\sigma^2)=\exp\left[\frac{y_i\theta-A(\theta)}{\sigma^2}+c(y_i,\sigma^2)\right] \tag{9.77}
$$
式中$\sigma^2$表示**散布参数（dispersion parameter）**（通常设置为1），$\theta$表示自然参数，$A$表示配分函数，$c$为正则化常数。举例来说，在逻辑回归中，$\theta$表示对数几率，$\theta=\log(\frac{\mu}{1=\mu})$，其中$\mu=\mathbb{E}[y]=p(y=1)$表示期望参数（见9.2.2.1）。为了将期望参数转换成自然参数，我们可以使用一个函数$\Psi$，所以$\theta=\Psi(\mu)$。这个函数由指数族的分布的形式唯一确定。事实上，这是一个可逆的映射，所以我们有$\mu=\Psi^{-1}(\theta)$。更进一步，我们在9.2.3节中了解到期望是由配分函数的导数决定的，所以我们有$\mu=\Psi^{-1}(\theta)=A^\prime(\theta)$。

现在让我们添加输入/协变量。我们首先定义一个关于输入的线性函数：
$$
\eta_i=\mathbf{w}^T\mathbf{x}_i \tag{9.78}
$$
通过某个可逆的单调函数，将上述的线性组合映射到分布的期望。按照惯例，这个函数被称为**期望函数（mean function）**，定义为$g^{-1}$，所以：
$$
\mu_i=g^{-1}(\eta_i)=g^{-1}(\mathbf{w}^T\mathbf{x}_i) \tag{9.79}
$$
图9.1给出了基本模型的示意图。

![avatar](../PIC/9.1.png)

$$
\mathbf{Figure\ 9.1} \ GLM的结构示意图。（图形来自于原书）
$$
期望函数的逆函数$g()$被称为**连接函数(link function)**。关于$g$的选择可以是任意的，只要它是可逆的，且它的反函数$g^{-1}$具备合适的值域。举例来说，在逻辑回归中，我们设置$\mu_i=g^{-1}(\eta_i)=sigm(\eta_i)$.

连接函数具有一种简单的形式，即$g=\Psi$，这被称为**典范连接函数(canonical link function)**。在这种情况下，$\theta_i=\eta_i=\mathbf{w}^T\mathbf{x}_i$，所以模型的形式为：
$$
p(y_i|\mathbf{x}_i,\mathbf{w},\sigma^2)=\exp\left[\frac{y_i\mathbf{w}^T\mathbf{x}_i-A(\mathbf{w}^T\mathbf{x}_i)}{\sigma^2}+c(y_i,\sigma^2)\right] \tag{9.80}
$$
在表9.1中，我们列出了一些分布和它们的典范连接函数。我们发现对于伯努利分布而言，典范连接函数为logit函数，$g(\mu)=\log(\frac{\eta}{1-\eta})$，其反函数为逻辑斯特函数，$\mu=sigm(\eta)$。

| 分布                        | 连接函数$g(\mu)$ | $\theta=\psi(\mu)$               | $\mu=\psi^{-1}(\theta)=\mathbb{E}[y]$ |
| --------------------------- | ---------------- | -------------------------------- | ------------------------------------- |
| $\mathcal{N}(\mu,\sigma^2)$ | $identity$       | $\theta=\mu$                     | $\mu=\theta$                          |
| $Bin(N,\mu)$                | $loghit$         | $\theta=\log(\frac{\mu}{1-\mu})$ | $\mu=sigm(\theta)$                    |
| $Poi(\mu)$                  | $log$            | $\theta=\log(\mu)$               | $\mu=e^\theta$                        |

$$
\mathbf{Table\ 9.1}\ 常规GLMs的典范连接函数\psi和它们的反函数
$$

基于9.2.3的介绍，我们可以知道响应变量的期望和方差，结果如下：
$$
\begin{eqnarray*}
\mathbb{E}[y|\mathbf{x}_i,\mathbf{w},\sigma^2] & = & \mu_i & = & A^\prime(\theta_i) \tag{9.81} \\
var[y|\mathbf{x}_i,\mathbf{w},\sigma^2] & = & \sigma^2 & = & A^{\prime\prime}(\theta_i)\sigma^2 \tag{9.82} \\
\end{eqnarray*}
$$
为了使符号更加清晰，让我们考虑某个简单的例子。

- 对于线性回归，我们有：
  $$
  \log p(y_i|\mathbf{x}_i,\mathbf{w},\sigma^2)=\frac{y_i\mu_i-\frac{\mu_i^2}{2}}{\sigma^2}-\frac{1}{2}\left(\frac{y_i^2}{\sigma^2}+\log(2\pi\sigma^2)\right) \tag{9.83}
  $$
  其中$y_i \in\mathbb{R}$，$\theta_i=\mu_i=\mathbf{w}^T\mathbf{x}_i$。其中$A(\theta)=\theta^2/2$，所以$\mathbb{E}[y_i]=\mu_i,var[y_i]=\sigma^2$.

- 对于二项式回归，我们有：
  $$
  \log p(y_i|\mathbf{x}_i,\mathbf{w})=y_i\log\left(\frac{\pi_i}{1-\pi_i}\right)+N_i\log(1-\pi_i)+\log\dbinom{N_i}{y_i} \tag{9.84}
  $$
  其中$y_i\in\{0,1,...,N_i\},\pi_i=sigm(\mathbf{w}^T\mathbf{x}_i),\theta_i=\log(\pi_i/(1-\pi_i))=\mathbf{w}^T\mathbf{x}_i,\sigma^2=1.$此处，$A(\theta)=N_i\log(1+e^\theta)$,所以$\mathbb{E}[y_i]=N_i\pi_i=\mu_i,var[y_i]=N_i\pi_i(1-\pi_i)$.

- 对于**泊松回归（poisson regression）**，我们有：
  $$
  \log p(y_i|\mathbf{x}_i,\mathbf{w})=y_i\log\mu_i-\mu_i-\log(y_i!) \tag{9.85}
  $$
  其中$y_i\in\{0,1,2,...\}，\mu_i=\exp(\mathbf{w}^T\mathbf{x}_i),\theta_i=\log(\mu_i)=\mathbf{w}^T\mathbf{x}_i,\sigma^2=1.$此处，$A(\theta)=e^\theta$,所以$mathbb{E}[y_i]=var[y_i]=\mu_i$。泊松回归在生物统计应用领域很广泛，其中$y_i$可能表示某个人或者地方的疾病数量，或者在高通量测序环境中基因组位置的读数。

###  9.3.2 ML和MAP估计

GLMs吸引人的性质在于它们可以使用相同的方法进行训练，这种方法与我们在逻辑回归中使用的方法一致。特别地，对数似然函数具备如下的形式：
$$
\begin{eqnarray*}
l(\mathbf{w})=\log p(\mathcal{D}|\mathbf{w}) & = & \frac{1}{\sigma^2}\sum_{i=1}^N
l_i \tag{9.86} \\
l_i & = & \theta_iy_i-A(\theta_i) \tag{9.87}
\end{eqnarray*}
$$

我们可以使用如下的链式法则计算梯度向量：
$$
\begin{align}
\frac{dl_i}{dw_j} & =\frac{dl_i}{d\theta_i}\frac{d\theta_i}{d\mu_i}\frac{d\mu_i}{d\eta_i}\frac{d\eta_i}{dw_j} \tag{9.88}\\
& = (y_i-A^\prime(\theta_i))\frac{d\theta_i}{d\mu_i}\frac{d\mu_i}{d\eta_i}x_{ij} \tag{9.89} \\
& = (y_i-\mu_i)\frac{d\theta_i}{d\mu_i}\frac{d\mu_i}{d\eta_i}x_{ij}\tag{9.90} \\
\end{align}
$$
如果我们使用典范连接函数，$\theta_i=\eta_i$，上式将简化为:
$$
\nabla_\mathbf{w}l(\mathbf{w})=\frac{1}{\sigma^2}\left[\sum_{i=1}^N(y_i-\mu_i)\mathbf{x}_i\right] \tag{9.91} \\
$$
上式表示输入向量的加权和，权重为误差。上式可以被用于随机梯度下降的算法中，这一点在8.5.2节讨论过。然而，为了提高效率，我们应该使用一个二阶方法。如果我们使用一种典范连接函数，海森矩阵由下式决定：
$$
\mathbf{H}=-\frac{1}{\sigma^2}\sum_{i=1}^N\frac{d\mu_i}{d\theta_i}\mathbf{x}_i\mathbf{x}_i^T=-\frac{1}{\sigma^2}\mathbf{X}^T\mathbf{S}\mathbf{X} \tag{9.92}
$$
其中$\mathbf{S}=diag(\frac{d\mu_1}{d\theta_1},...,\frac{d\mu_N}{d\theta_N})$为对角权重矩阵。上式可以在IRLS算法（8.3.4节）中使用。特别地，我们可以使用如下的牛顿更新方法：
$$
\begin{eqnarray*}
\mathbf{w}_{t+1} & = & (\mathbf{X}^T\mathbf{S}_t\mathbf{X})^{-1}\mathbf{X}^T\mathbf{S}_t\mathbf{z}_t \tag{9.93} \\
\mathbf{z}_t & = & \mathbf{\theta}_t+\mathbf{S}_t^{-1}(\mathbf{y-\mathbf{\mu}_t}) \tag{9.94}
\end{eqnarray*}
$$
其中$\mathbf{\theta}_t=\mathbf{X}\mathbf{w}_t,\mathbf{\mu}_t=g^{-1}(\mathbf{\eta}_t)$。

如果我们将上述推导拓展到非典范形式的连接函数，我们将会发现海森矩阵具备额外项。然而，结果表明期望海森的形式与式9.92是一样的；使用期望海森（又被称为费舍尔信息矩阵）而不是实际的海森矩阵，这种方法被称为**费舍尔评价方法（Fisher scoring method）.**

将上述程序应用于含高斯先验的MAP估计也是可以的：我们只需要调整目标函数，梯度和海森，就像在8.3.6节介绍的那样，为逻辑回归添加一个$l_2$正则项。

### 9.3.3 贝叶斯推理

GLMs的贝叶斯推理一般采用MCMC方法（第24章）。其他方法包括<font color=#00ffff>**待定**</font>

## 9.4 Probit 回归

在二分类逻辑回归中，我们使用的模型形式为$p(y=1|\mathbf{x}_i,\mathbf{w})=sigm(\mathbf{w}^T\mathbf{x}_i)$。一般情况下，我们可以写成$p(y=1|\mathbf{x}_i,\mathbf{w})=g^{-1}(\mathbf{w}^T\mathbf{x}_i)$，其中$g^{-1}$为将区间$[-\infty,\infty]$映射到$[0,1]$。几种可能的期望函数如表9.2所示。

| Name                  | Fromula                                           |
| --------------------- | ------------------------------------------------- |
| 逻辑斯特              | $g^{-1}(\eta)=sigm(\eta)=\frac{e^\eta}{1+e^\eta}$ |
| Probit                | $g^{-1}(\eta)=\Phi(\eta)$                         |
| Log-log               | $g^{-1}(\eta)=\exp(-\exp(-\eta))$                 |
| Complementary log-log | $g^{-1}(\eta)=1-\exp(-\exp(\eta))$                |

$$
\mathbf{Table\ 9.2} \ 在二值回归中可能用到的期望函数
$$

本节，我们将集中讨论$g{-1}(\eta)=\Phi(\eta)$的情况，其中$\Phi(\eta)$为标准正态分布累积分布函数。这被称为**probit 回归**。probit函数与逻辑斯特函数十分相似。然而，前者相较于后者却有一些优势，关于这一点我们将会在下文看到。

### 9.4.1 使用基于梯度优化的方法求解ML/MAP估计

我们可以使用标准的梯度方法求解probit回归模型的MLE。令$\mu_i=\mathbf{w}^T\mathbf{x}_i，\tilde{y_i}\in\{-1,+1\}​$。对于某一个样例而言，对数似然函数的梯度为：
$$
\mathbf{g}_i=\frac{d}{d\mathbf{w}}\log p(\tilde{y_i}|\mathbf{w}^T\mathbf{x}_i)=\frac{d\mu_i}{d\mathbf{w}}\frac{d}{d\mu_i}\log p(\tilde{y_i}|\mathbf{w}^T\mathbf{x}_i)=\mathbf{x}_i\frac{\tilde{y_i}\phi(\mu_i)}{\Phi(\tilde{y_i}\mu_i)} \tag{9.95}
$$
其中$\phi$为标准正态分布的概率密度函数，$\Phi$为它的累计密度函数。类似地，对于单个样例，其海森矩阵为：
$$
\mathbf{H}_i=\frac{d}{d\mathbf{w}^2}\log p(\tilde{y_i}|\mathbf{w}^T\mathbf{x}_i)=-\mathbf{x}_i\left(\frac{\phi(\mu_i)^2}{\Phi(\tilde{y_i}\mu_i)^2}+\frac{\tilde{y_i}\mu_i\phi(\mu_i)}{\Phi(\tilde{y_i}\mu_i)}\right)\mathbf{x}_i^T\tag{9.96}
$$
为了计算MAP估计，我们可以对上面的方式进行直接的调整。特别地，如果我们使用先验分布$p(\mathbf{w})=\mathcal{N}(\mathbf{0},\mathbf{V}_0)$，含惩罚项的对数似然函数的梯度和海森矩阵的形式为$\sum_{i}\mathbf{g}_i+2\mathbf{V}_0^{-1}\mathbf{w}$和$\sum_i\mathbf{H}_i+2\mathbf{V}_0^{-1}$。这些表达式可以用于任意的基于梯度的优化方法。程序**probitRegDemo**给出了案例。

### 9.4.2 潜在变量解释

我们可以对probit（和逻辑斯特）模型作出如下的解释。首先，让我们将每一个输入项$\mathbf{x}_i$与两个潜在效能$u_{0i},u_{1i}$关联，分别对应$y_i=0$和$y_i=1$的选择。然后我们假设观察到的选择对应于那个效能比较大的选项。更加精确的表述如下：
$$
\begin{eqnarray*}
u_{0i} & = & \mathbf{w}_0^T\mathbf{x}_i+\delta_{0i} \tag{9.97} \\
u_{1i} & = & \mathbf{w}_1^T\mathbf{x}_i+\delta_{1i} \tag{9.98} \\
y_i & = & \mathbb{I}(u_{1i}\gt u_{0i}) \tag{9.99} 
\end{eqnarray*}
$$
其中我们令$\mathbf{w}=\mathbf{w}_1-\mathbf{w}_0$,$\delta$为误差项，表征了那些影响最终决策但并未包含（或无法包含）在模型中的因素。这被称为**随机效能模型（random utility model,RUM）**。

既然影响结果的只有效能，那么让我们定义$z_i=u_{1i}-u_{0i}=\mathbf{w}^T\mathbf{x}_i+\epsilon_i$，其中$\epsilon_i=\delta_{1i}-\delta_{0i}$，如果$\delta$服从高斯分布，那么$\epsilon_i$也服从高斯分布。这样我们就有：
$$
\begin{eqnarray*}
z_i & = & \mathbf{w}^T\mathbf{x}_i+\epsilon_i \tag{9.100} \\
\epsilon_i & \sim & \mathcal{N}(0,1) \tag{9.101} \\
y_i = 1 & = & \mathbb{I}(z_i \ge 0) \tag{9.102} \\
\end{eqnarray*}
$$
我们称这个模型为差分RUM或者$\mathbf{dRUM}$。

当我们对$z_i$进行积分，我们将得到probit模型：
$$
\begin{align}
p(y_i=1|\mathbf{x}_i,\mathbf{w}) & =  \int{}\mathbb{I}(z_i \ge 0)\mathcal{N}(z_i|\mathbf{w}^T\mathbf{x}_i,1)dz_i \tag{9.103} \\
& =p(\mathbf{w}^T\mathbf{x}_i+\epsilon_i \ge 0) =p(\epsilon_i \ge -\mathbf{w}^T\mathbf{x}_i) \tag{9.104}  \\
& = 1-\Phi(-\mathbf{w}^T\mathbf{x}_i) = \Phi(\mathbf{w}^T\mathbf{x}_i) \tag{9.105}
\end{align}
$$
此处我们应用了高斯分布的对称性。这种潜在变量的解释提供了训练模型的另一种方式，我们在11.4.6节进行讨论。

有趣的是，如果我们使用对$\delta$使用Gumbel分布，那么$\epsilon_i$将服从罗基斯特分布，模型将退化成逻辑回归。更多细节见24.5.1节。

### 9.4.3 顺序probit回归*

关于probit回归潜变量解释的一个优势在于它很容易地拓展到其他应用场景，在这种场景中，响应变量是有序的，也就是说目标变量的取值范围是C个可排序的离散值，比如说：低、中、高。这种场景的模型被称作**顺序回归（ordinal regression）**。其基本思想如下。我们引入C+1个阈值$\gamma_j$，并且令
$$
y_i=j \qquad if \qquad \gamma_{j-1} \lt z_i \le \gamma_{j} \tag{9.106}
$$
其中$\gamma_0 \le ... \le \gamma_C$。出于可辨识性的原因，我们设置$\gamma_0 = -\infty,\gamma_1=0,\gamma_C=\infty$，如果$C=2$，上式将退化成标准的二值probit模型，当$z_i \lt 0$，得到$y_i=0$；$z_i \ge 0$,得到$y_i = 1$。如果$C=3$，我们将实现分成三个区间:$(-\infty,0],(0,\gamma_2],(\gamma_2,\infty)$。通过调整$\gamma_2$确保在不同区间内的相对概率质量是合理的，也就是说与每个类别标签下的经验频率匹配。

相较于二值probit回归，找到该模型的MLEs是一件麻烦的事情，因为我们需要对$\mathbf{w}$和$\mathbf{\gamma}$进行优化，后者必须满足有序的约束。

### 9.4.4 Multinomial probit回归*

现在考虑一种情况，其中响应变量的取值为$C$个无序的类别值，即$y_i \in \{1,...,C\}$。$\mathbf{multinomial\quad probit}$模型定义为：

待定

## 9.5 多任务学习

有些情况下，我们需要训练许多相关的分类或者回归模型。一般情况下，假设输入与输出的映射关系在不同模型之间是相似的，所以同时对所有参数精心训练可以获得更好的性能。在机器学习领域，这个场景被称为**多任务学习（multi-task learning）**，**迁移学习(transfer learning)**，**学习学习(learning to learn)**。在统计领域，通常使用分层贝叶斯模型来解决这类问题，这一点我们将在下面介绍，尽管还有其他的方法，比如高斯过程。

### 9.5.1 多任务学习的分层贝叶斯方法









