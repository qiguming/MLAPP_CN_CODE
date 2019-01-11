from preprocessorCreate import polyX, rescaleX
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from CvEstimate import cv_estimate
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
# 产生训练集和测试集
n_samples = 21
xtrain = np.linspace(0,20,n_samples)[:,np.newaxis]
xtest = np.arange(0,20,0.1)[:,np.newaxis]
w = np.array([-1.5, 1/9])
fun = lambda x:w[0]*x + w[1]*x**2
sigma = 4

ytrain = fun(xtrain) + np.random.randn(len(xtrain),1)*sigma**0.5

ytestNoisefree = fun(xtest)
ytestNoisey = ytestNoisefree + np.random.randn(len(xtest),1)*sigma**0.5

# 对数据y进行归一化，此处对y进行求均值处理，最终的影响只是在y轴上的上下平移
ytrain = ytrain - np.mean(ytrain)
ytest = ytestNoisey - np.mean(ytestNoisey)

# 对数据进行多维拓展和归一化
pp_poly = polyX(degree=14)
pp_resc = rescaleX()
pp_list = [pp_resc, pp_poly]
Xtrain = xtrain.copy()
Xtest = xtest.copy()
# 对训练集和测试集应该采用同样的预处理方式
for pp in pp_list:
    Xtrain = pp.preprocess(Xtrain)
    Xtest = pp.preprocess(Xtest)

# 导入线性回归模型
lr = linear_model.LinearRegression()
lr.fit(Xtrain, ytrain)
print(lr.coef_[0])
#print("{:.3f}".format(lr.coef_[0]))
print(list("{0:.3f}".format(coff) for coff in lr.coef_[0]))

# 对测试集进行预测
ypredTest = lr.predict(Xtest)

# 绘图
plt.scatter(xtrain, ytrain, color='blue')
plt.plot(xtest, ypredTest, color='black', linewidth=3)
plt.show()

### 采用岭回归

lambdas = np.logspace(-10, 1.3, 10)
NL = len(lambdas)
printNdx = [1, 5]
testMse = np.zeros(NL)
trainMse = np.zeros(NL)

for k in range(NL):
    _lambda = lambdas[k]
    ridge_lr = linear_model.Ridge(alpha=_lambda)
    ridge_lr.fit(Xtrain, ytrain)
    ypredTest = ridge_lr.predict(Xtest)
    ypredTrain = ridge_lr.predict(Xtrain)
    
    testMse[k] = np.mean(((ypredTest-ytest).flatten())**2)
    trainMse[k] = np.mean(((ypredTrain-ytrain).flatten())**2)

ndx = np.log10(lambdas)
plt.plot(ndx, trainMse, 'bs:', linewidth=2, markersize=12, label='train')
plt.plot(ndx, testMse, 'rx-', linewidth=2, markersize=12, label='test')
plt.xlabel('log10 lambda')
plt.legend()
plt.show()


# 打印两张岭回归后的图
for k in printNdx:
    _lambda = lambdas[k]
    print(np.log10(_lambda))
    ridge_lr = linear_model.Ridge(alpha = _lambda)
    ridge_lr.fit(Xtrain, ytrain)
    print(list("{0:.3f}".format(coff) for coff in ridge_lr.coef_[0]))
    ypredTest = ridge_lr.predict(Xtest)
    ypredTrain = ridge_lr.predict(Xtrain)
    modelSigm = np.sum((ypredTrain-ytrain)**2)/ytrain.shape[0]  # 模型预测的方差
    plt.scatter(xtrain, ytrain, color='black')
    plt.plot(xtest, ypredTest, color='k', linewidth=3)
    plt.plot(xtest, ypredTest-modelSigm, 'b:', linewidth=3)
    plt.plot(xtest, ypredTest+modelSigm, 'b:', linewidth=3)
    plt.title('lambda={:.2}'.format(np.log10(_lambda)))
    plt.show()

###  交叉验证
mse = []
for k in range(NL):
    _lambda = lambdas[k]
    ridge_lr = linear_model.Ridge(alpha=_lambda)
    nfolds = 5
    scores = cross_validate(ridge_lr, Xtrain, ytrain.flatten(), scoring='neg_mean_squared_error', cv=5)
    mse.append(-np.mean(scores['test_score']))
new_mse = np.log(mse)
plt.plot(np.log10(lambdas), new_mse, 's-', color='b', linewidth=3, markersize=8, markeredgewidth=3, markerfacecolor='white')
plt.show()












