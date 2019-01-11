# 该程序实现学习曲线
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(10,7)
from sklearn import preprocessing, linear_model, metrics
def sample_data(n):
    """采样生成数据集"""
    xtrain = np.linspace(0, 20, n)[:,np.newaxis]
    xtest = np.arange(0, 20, 0.1)[:, np.newaxis]
    sigma2 = 4
    w = [-1.5, 1/9]
    fun = lambda x: w[0]*x + w[1]*(x**2)

    ytrain = fun(xtrain) + np.random.randn(np.size(xtrain),1)*(sigma2**0.5)
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + np.random.randn(np.size(xtest),1)*(sigma2**0.5)

    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2

degrees = [1,2,10,25]
xlabel =['(a)','(b)', '(c)', '(d)']
for index, degree in enumerate(degrees):
    ns = np.linspace(10, 200, 10)
    testMse = np.zeros(len(ns))
    trainMse = np.zeros(len(ns))

    for i,n in enumerate(ns):
        xtrain, ytrain, xtest, ytestNoisefree, ytest, sigma2 = sample_data(n)
        # 对数据进行预处理，将其处理到[-1,1]区间
        max_abs_scaler = preprocessing.MaxAbsScaler()
        xtrain = max_abs_scaler.fit_transform(xtrain)
        xtest = max_abs_scaler.transform(xtest)
        poly = preprocessing.PolynomialFeatures(degree=degree)
        xtrain = poly.fit_transform(xtrain)
        xtest = poly.transform(xtest)
        # 对模型进行训练
        model = linear_model.LinearRegression()
        model.fit(xtrain, ytrain)
        ytrainhat = model.predict(xtrain)
        trainMse[i] = metrics.mean_squared_error(ytrainhat, ytrain)
        ytesthat = model.predict(xtest)
        testMse[i] = metrics.mean_squared_error(ytesthat, ytest)
    
    ax = plt.subplot(2,2,index+1)
    ax.plot(ns, trainMse, 'bs--', label='train')
    ax.plot(ns, testMse, 'rx-', label='test')
    ax.hlines(sigma2, np.min(ns), np.max(ns))
    ax.set_ylim((0,22))
    ax.set_xlim((np.min(ns), np.max(ns)))
    ax.set_title('truth=degree 2, model=degree {}'.format(degree))
    ax.set_xlabel(xlabel[index])
    ax.legend()
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=0.2, hspace=0.3)
plt.show()
