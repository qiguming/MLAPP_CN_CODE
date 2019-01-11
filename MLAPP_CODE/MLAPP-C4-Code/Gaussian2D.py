import numpy as np
from scipy import stats
def gaussian_2d(mean, cov):
    """
    绘制2维高斯分布的轮廓线
    Input：
    mean: 期望；shape=(2,)
    cov： 协方差矩阵, shape=(2,2)
    Output：
    z:   轮廓线上的坐标, shape=(2,n_points)
    """
    n = 200                            # 绘图需要的点的数量
    eigvals, U = np.linalg.eig(cov)    # 矩阵U表示由特征向量组成的正交矩阵
    D = np.diag(eigvals)               # 由特征值构造出来的对角矩阵D
    k = np.power(stats.chi2.ppf(0.95, 2), 0.5)    # 绘制出95%概率质量覆盖的轮廓线，此处使用了卡方分布的相关知识
    t = np.linspace(0, 2*np.pi, n)
    xy = np.array([np.cos(t), np.sin(t)])
    w = np.dot(np.dot(k*U, np.power(D, 0.5)), xy)
    z = w + mean[:, np.newaxis]        # z为%95概率质量的轮廓线上的坐标          
    return z