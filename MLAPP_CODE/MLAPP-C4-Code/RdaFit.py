import numpy as np
from functools import reduce
def rda_fit(model, X, y, _lambda, R, V):
    """实现RDA模型的训练
    Input:
    model: 判别式模型类
    X: 设计矩阵, shape=(n_samples, dim)  note: dim > n_samples
    y: 类别标签, shape=(n_samples,)
    _lambda: 正则化系数
    R: 低维空间的设计矩阵， shape=(n_samples,n_samples)
    V: 右奇异矩阵,   shape=(dim,n_samples)
    """
    if not R:
        U, S, V = np.linalg.svd(X, full_matrices=False)
        R = np.dot(U, np.diag(S))
    n_classes = len(np.unique(y))
    dim = X.shape[1]
    R_cov = np.cov(R, rowvar=False)   # 计算低维设计矩阵的协方差矩阵, rowvar=False表示每一列代表一个变量
    S_reg = _lambda*R_cov + (1-_lambda)*np.diag(np.diag(R_cov))
    S_inv = np.linalg.inv(S_reg)
    model.beta = np.zeros((n_classes, dim))
    for i in range(n_classes):
        index = (y==i)
        muRed = np.mean(R[index], axis=0)[:,np.newaxis]
        model.beta[i, :] = reduce(np.dot, (V, S_inv, muRed))
    return model