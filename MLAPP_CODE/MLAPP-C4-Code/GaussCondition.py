import numpy as np
from collections import Iterable
from functools import reduce
def gauss_condition(model, v, vis_value):
    """在多维高斯分布，已知部分维度v中的数据vis_value，求取其他维度的
    条件概率分布
    Input:
    model: 高斯模型类，含期望(mu)和协方差矩阵(sigma)两个属性
    v(visual):     已知观测值的特征所在的维度, shape=(n_v,)/single_int
    vis_value: 对应维度的观测值, shape=(n_v,)
    Output:
    mu_v:    条件期望
    sigma_v: 条件协方差矩阵
    """
    mu = model.mu; sigma = model.sigma
    dim = len(mu)
    if not isinstance(v, Iterable):
        v = np.array([v])                          
    else:
        v = np.array(v)
    h = np.array(list(set(np.arange(dim))-set(v)))  # 那些未给定观测值的维度(hidden)
    if len(h) ==0:   # 如果h为空，即所有维度都给定观测值了
        mu_v = []; sigma_v = []
    elif len(v) ==0:  # 如果v为空
        mu_v = mu; sigma_v = sigma
    else:
        h = h[:, np.newaxis]; v = v[:, np.newaxis]
        s_hh = sigma[h, h.T]; s_hv =sigma[h, v.T]; s_vv = sigma[v, v.T]   #  花式索引
        s_vv_inv = np.linalg.inv(s_vv)
        mu_v = mu[h.flatten()] + (reduce(np.dot, (s_hv, s_vv_inv, (vis_value-mu[v.flatten()])[:, np.newaxis]))).flatten()
        sigma_v = s_hh - reduce(np.dot,(s_hv, s_vv_inv, s_hv.T))
    return mu_v, sigma_v
    