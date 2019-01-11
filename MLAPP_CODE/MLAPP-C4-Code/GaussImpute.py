import numpy as np
from GaussCondition import gauss_condition
def gauss_impute(model, X):
    """
    基于高斯分布进行数据重构
    Input:
        model:  包含属性mu和sigma的类
        X:       含缺失值的设计矩阵
    Output:
        Xc[i, j] = E[X[i, j]|D]
        V[i, j] = Var
    """
    n_samples, dim = X.shape
    Xc = X.copy()
    V = np.zeros((n_samples, dim))

    for i in range(n_samples):
        hid_index = np.where(np.isnan(X[i, :]))[0]
        if len(hid_index)==0: continue
        vis_index = np.where(~ np.isnan(X[i, :]))[0]
        vis_value = X[i, vis_index]
        mu_v, sigma_v = gauss_condition(model, vis_index, vis_value)
        Xc[i, hid_index] = mu_v
        V[i, hid_index] = np.diag(sigma_v)
    return Xc, V

