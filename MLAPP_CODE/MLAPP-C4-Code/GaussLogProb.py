import numpy as np
# import scipy.stats as stats
def gauss_log_prob(mu, Sigma, X):
    """计算多变量高斯分布的对数概率密度
    Input:
    X: shape=(n_samples, dim)
    mu: shape=(dim, )
    Sigma: shape=(dim, dim)
    Output:
    logp：对数概率密度
    """
    if len(Sigma.shape) == 1 and len(Sigma) > 1:   # 如果是对角矩阵
        Sigma = np.diag(Sigma)
    dim = X.shape[1]
    X = X - mu[np.newaxis, :]                    # 将mu转换为shape=(1,d)
    R = np.linalg.cholesky(Sigma)                # 矩阵cholesky分解
    log_p = -0.5*np.sum(np.power(np.dot(X, np.linalg.inv(R)), 2), 1)
    log_z = 0.5*dim*np.log(2*np.pi) + np.sum(np.log(np.diag(R)))
    log_p = log_p - log_z
    # logp = stats.multivariate_normal.logpdf(X, mu, Sigma)    # 我们也可以直接使用scipy库中的函数直接生成对数概率
    return log_p