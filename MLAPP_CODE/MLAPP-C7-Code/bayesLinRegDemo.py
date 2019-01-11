import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# 真实的参数值
a0 = -0.3;a1 = 0.5

trainingPoints = 100   # 训练样本的数量
noiseSD = 0.2          # 高斯噪音的标准差
priorPrecision = 2.0   # 先验分布的精度
likelihoodSD = noiseSD # 假设似然函数的精度已知
likelihoodPrecision = 1/(likelihoodSD)**2

# 产生训练样本
xtrain = -1 + 2*np.random.rand(trainingPoints, 1)
noise = stats.norm.rvs(loc=0, scale=noiseSD, size=trainingPoints)[:, np.newaxis]
ytrain = a0 + a1*xtrain + noise

def contourPlot(func, trueValue):
    numSteps = 200
    xyrange = np.linspace(-1,1,numSteps)
    [x, y] = np.meshgrid(xyrange, xyrange)
    xy = np.dstack((x,y)).reshape(-1,2)
    p = func(xy)
    p = p.reshape(x.shape)

    plt.contourf(x, y, p, cmap=plt.get_cmap("jet"))
    plt.axis('square')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('W0')
    plt.ylabel('W1')

    if len(trueValue) == 2:
        plt.plot(trueValue[0], trueValue[1], '+w')

def plotSampleLines(mean, sigma, numlines, dataPoints):
    for _ in range(numlines):
        w = stats.multivariate_normal(mean, sigma).rvs(size=1)
        func = lambda x: w[0] + w[1]*x
        x_range = np.linspace(-1,1,20)
        y_range = func(x_range)
        plt.plot(x_range, y_range, 'r')
    plt.axis('square')
    plt.xlim([-1,1])
    plt.ylim([-1,1])

def update(xtrain, ytrain, likelihoodPrecision, priorMean, priorSigma):
    postSigma = np.linalg.inv(np.linalg.inv(priorSigma) + likelihoodPrecision*((xtrain.T).dot(xtrain)))
    postMu = postSigma.dot(np.linalg.inv(priorSigma)).dot(priorMean) + likelihoodPrecision*postSigma.dot(xtrain.T).dot(ytrain)
    postW = lambda W:stats.multivariate_normal(postMu, postSigma).pdf(W)
    return postW




# 绘制图形
iter = 2
ax = plt.subplot(iter+2, 3, 2)
priorMean = [0, 0]
priorSigma = np.eye(2)/priorPrecision   # 协方差矩阵
priorPDF = lambda w:stats.multivariate_normal(priorMean, priorSigma).pdf(w)
contourPlot(priorPDF, [])

ax = plt.subplot(iter+2,3,3)
plotSampleLines(priorMean, priorSigma, 6, [])

for i in range(iter):
    ax = plt.subplot(2+iter,3,3*i+4)
    xtrain_new = np.array([1,xtrain[i]]).reshape((2,1))
    likelihood = lambda W: stats.multivariate_normal(mean=W.dot(xtrain_new), cov=)

plt.show()
