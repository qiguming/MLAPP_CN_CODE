import numpy as np
from CanonizeLabels import canonize_labels
from ShrunkenCentroidsFit import  shrunken_centroids_fit
from RdaFit import rda_fit


def discrim_analysis_fit(X, y, model_type, **varargin):
    """
    该函数实现一个判别式模型的训练
    Input:
    X: 样本组成的设计矩阵，shape=(n_samples,dim)
    y: 样本的标签值，      shape=(n_samples,)
    model_type:
          1.'linear'(共享协方差矩阵)
          2.'quadratic'(类相关的协方差矩阵)
          3.'RDA'(正则化线性判别分析)
          4.'diag'(对角二次判别分析——对应朴素贝叶斯)
          5.'shrunkenCentroids'(含L1正则项的对角线性判别分析)
    如果使用RDA或者shrunkenCentroids，必须给定参数lambda。
    Output:
    model.model_type:         模型的种类，            str
    model.class_prior:        模型的类先验分布，       shape=(n_classes,)
    model.n_classes:          模型的类的数量，         int
    model._lambda:            用于控制正则化的系数，   float
    model.mu:                 模型期望值，            shape=(n_classes,dim)
    满足如下情况时，同时返回
        如果type==QDA,    model.sigma:          协方差矩阵，           shape=(n_classes,dim,dim)
        如果type==LDA,    model.sigma_pooled:   共享协方差矩阵，       shape=(dim,dim)
        如果type==RDA,    model.beta:                                 shape=(dim,)
        如果type==diag,   model.sigma_diag:     对角协方差矩阵,        shape=(n_classes,dim)
        如果type==shrunkenCentroids, model.sigma_pooled_diag            shape=(n_classes,)
                                     model.shrunken_centroids,         shape=(n_classes,dim)
    """
    class DiscrimModel():                           # 定义判别式模型类
        """
        定义一个判别式模型类，此处我们不定义实例的属性，因为不同的判别式
        模型所需要的属性并不一样
        """
        def __init__(self):
            pass

    _lambda = varargin.get('lambdas', [])           # 获取正则项系数
    R = varargin.get('R', [])                       # 用于RDA模型中的低维设计矩阵
    V = varargin.get('V', [])
    pseudo_count = varargin.get('pseudo_count', 1)   # 获取伪计数

    model = DiscrimModel()                          # 实例化判别式模型
    model._lambda = _lambda
    model.model_type = model_type

    y, model.support = canonize_labels(y)            # 返回重新调整后的类别标签y和支撑集support
    model.n_classes = len(model.support)             # 类的数量
    n_samples, dim = X.shape                         # 获取样本数量和样本维度

    model.mu = np.zeros((model.n_classes, dim))      # 初始化类别期望
    model.class_prior = np.zeros((model.n_classes,)) # 初始化类先验分布
    x_bar = np.mean(X, axis=0)                       # 初始化与类别无关的期望，此处称为全局期望
    ns_per_class = np.zeros((model.n_classes,))      # 初始化每个类中样本的数量(n_samples_per_class)

    for k in range(model.n_classes):
        index_k = (y == k)                           # shape=(n_samples,1)
        ns_per_class[k] = np.sum(index_k.flatten())
        model.class_prior[k] = ns_per_class[k] + pseudo_count
        if ns_per_class[k] == 0:
            model.mu[k, :] = x_bar                     # 如果某个类别中不存在任何样本，则使用全局期望
        else:
            model.mu[k, :] = np.mean(X[index_k.flatten()], axis=0)
    model.class_prior = model.class_prior/np.sum(model.class_prior)  # 归一化类先验分布

    _model_type = str.lower(model_type)
    if _model_type in ['lda', 'linear']:              # 线性模型
        sigma_pooled = np.zeros((dim, dim))
        for c in range(model.n_classes):
            index = (y==c)
            n_c = np.sum(index)
            dat = X[index, :]
            sigma = np.cov(dat, rowvar=False)
            sigma_pooled = sigma_pooled + n_c*sigma   # 加权协方差矩阵
        model.sigma_pooled = sigma_pooled/n_samples
    elif _model_type in ['qda', 'quadratic']:                     # 二次判别分析
        model.sigma = np.zeros((model.n_classes, dim, dim))
        for c in range(model.n_classes):
            index = (y==c)
            dat = X[index]
            model.sigma[c, :, :] = np.cov(dat, rowvar=False)
    elif _model_type == 'diag':                                 # 对角协方差模型
        model.sigma_diag = np.zeros((model.n_classes, dim))
        for c in range(model.n_classes):
            index = (y==c)
            dat = X[index]
            model.sigma_diag[c, :] = np.var(dat, axis=0)
    elif _model_type == 'shrunkencentroids':
        model = shrunken_centroids_fit(model, X, y, _lambda)     # 对模型进行训练
    elif _model_type == 'rda':                                   # RDA 模型
        if not R:                                                # 如果未指定低维设计矩阵，则采用SVD进行计算
            U,S,V = np.linalg.svd(X, full_matrices=False)
            R = np.dot(U, np.diag(S))
        model = rda_fit(model, X, y, _lambda, R, V)
    
    
    
    else:
        raise Exception('不存在类型{}'.format(model_type))
    return model


