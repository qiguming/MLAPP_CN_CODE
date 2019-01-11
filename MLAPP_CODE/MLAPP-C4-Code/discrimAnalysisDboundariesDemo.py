import numpy as np
# from scipy import stats                          # 用于多变量高斯分布
import matplotlib.pyplot as plt                  # 用于绘图
from DiscrimAnalysisPredict import discrim_analysis_predict
# import gaussLogprob
from PlotDecisionBoundary import plot_decision_boundary
from MixGaussSample import mix_gauss_sample


#  定义判别式模型类
class discri_model:
    def __init__(self, class_prior, mu, sigma, model_type, sigma_pooled=None):
        self.class_prior = class_prior             # 类先验分布
        self.mu = mu                              # 期望
        self.sigma = sigma                       # 协方差矩阵
        self.model_type = model_type             # 模型的类型：线性还是二次型
        self.sigma_pooled = sigma_pooled           # 当模型为线性判别式时，使用共享协方差矩阵

#   实例化判别模型
#   二分类线性判别分析模型
class_prior = np.array([1, 1])/2                  # 类先验分布p(y=c),shape=(nclass,)
mu = np.array([[1.5, 1.5], [-1.5, -1.5]])        # 期望 shape=(class_num,dim)
sigma = np.tile(np.diag([1, 1]), [2, 1, 1])      # 协方差矩阵 shape=(class_num,dim,dim)
sigma_pooled = np.diag([1, 1])                    # 共享协方差矩阵
model_type = 'linear'                            # 指定模型类别
discri_model1 = discri_model(class_prior, mu, sigma, model_type, sigma_pooled=sigma_pooled)

#   二分类二次判别分析模型
class_prior = np.array([1, 1])/2                     # 类先验分布p(y=c),shape=(nclass,)
mu = np.array([[1.5, 1.5],[-1.5, -1.5]])            # 期望 shape=(class_num,dim)
cov1 = np.array([[1.5, 0], [0, 1]])
cov2 = 0.7*np.diag([1, 1])
sigma = np.vstack((cov1, cov2)).reshape(2, 2, 2)    # 协方差矩阵 shape=(class_num,dim,dim)
model_type = 'quadratic'                            # 指定模型类别
discri_model2 = discri_model(class_prior, mu, sigma, model_type)

#   三分类线性判别分析模型
class_prior = np.array([1, 1, 1])/3                      # 类先验分布p(y=c),shape=(nclass,)
mu = np.array([[0, 0], [0, 5], [5, 5]])                 # 期望 shape=(class_num,dim)
cov = np.diag([1, 1])
sigma = np.vstack((cov, cov, cov)).reshape(3, 2, 2)     # 协方差矩阵 shape=(class_num,dim,dim)
sigma_pooled = np.diag([1, 1])                          # 共享协方差矩阵
model_type = 'linear'                                   # 指定模型类别
discri_model3 = discri_model(class_prior, mu, sigma, model_type, sigma_pooled=sigma_pooled)

#   三分类混合模型
class_prior = np.array([1, 1, 1])/3                    # 类先验分布p(y=c),shape=(1,2)
mu = np.array([[0, 0], [0, 5], [5, 5]])               # 期望 shape=(class_num,dim)
cov1 = np.array([[4, 0], [0, 1]])
cov2 = 0.7*np.diag([1, 1])
sigma = np.vstack((cov1, cov2, cov2)).reshape(3, 2, 2)    # 协方差矩阵 shape=(class_num,dim,dim)
model_type = 'quadratic'                                  # 指定模型类别
discri_model4 = discri_model(class_prior, mu, sigma, model_type)

model=[discri_model1, discri_model2, discri_model3, discri_model4]

titles = ['Linear Boundary', 'Parabolic Boundary', 'All Linear Boundaries ', 'Some Linear, Some Quadratic']
# fnames = ['dboundaries2classLinear', 'dboundaries2classParabolic', 'dboundaries3classLinear', 'dboundaries3classParabolic']
n_samples = 30                                       # 设置采样的点的数目
c = ['b','r','g','c','y','m']
markers = [ '*', '+', 'x','s', 'p','h']
for i in range(len(model)):
    fig = plt.figure(figsize=(8, 7))
    [X, y] = mix_gauss_sample(model[i].mu, model[i].sigma, model[i].class_prior, n_samples)    #  根据每一个高斯分布参数进行采样
    # print(X, y)
    [X1, X2, log_lik] = plot_decision_boundary(X, y, lambda X_test:discrim_analysis_predict(model[i], X_test))
    for j in range(len(model[i].mu)):
        # XY = np.dstack((X1,X2)).reshape((-1,model[i].mu.shape[1]))
        # z = gaussLogprob(model[i].mu[j,:], model[i].sigma[j,:,:], XY).reshape(X1.shape)
        z = log_lik[:, j].reshape(X1.shape)
        plt.contour(X1, X2, np.exp(z), colors=c[j])
    plt.title(titles[i])
    plt.show()


