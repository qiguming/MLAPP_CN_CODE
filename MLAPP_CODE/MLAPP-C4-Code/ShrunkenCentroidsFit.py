import numpy as np
from SoftThreshold import soft_threshold
def shrunken_centroids_fit(model, X_train, y_train, _lambda):
    """对收缩质心模型进行训练
    关于该模型训练的详细原理可参考书籍
    《The Elements of Statistical Learning》(ESL)2nd editor,section 18.2
    Input:
    model:  DiscrimModel实例
    Xtrain: 设计矩阵,     shape=(n_samples, dim)
    ytrain: 类标签索引值, shape=(n_samples,)
    _lambda: 用于Cross-Validation
    Output:
    model
    """
    X_train = np.array(X_train)               # 将数据转换为numpy类型
    y_train = np.array(y_train)
    n_classes = len(np.unique(y_train))       # 类别的数量
    n_samples, dim = X_train.shape            # 获取样本的数量和每个样本的维度
    ns_per_class = np.empty((n_classes, ))    # 每个类中样本的数量

    # 计算混合标准差
    x_bar = np.mean(X_train, axis=0)          # shape = (dim,)
    s_error = np.zeros((dim, ))               # 初始化标准差
    for c in range(n_classes):
        index = (y_train == c)
        ns_per_class[c] = np.sum(index)       # 在类c中共有多少个样本
        # 如果在类c中不存在样本，则使用均值x_bar作为该类分布的质心
        if ns_per_class[c] == 0:
            centroid = x_bar
        else:
            centroid = np.mean(X_train[index.flatten()], axis=0)
        temp1 = X_train[index.flatten()]
        temp2 = centroid[np.newaxis, :]
        temp3 = np.power(temp1 - temp2, 2)
        s_error = s_error + np.sum(temp3, axis=0)
    sigma = np.power(s_error/(n_samples - n_classes), 0.5)   # 混合标准差,shape=(dim,)
    s0 = np.median(sigma)                                    # 中位数

    mu = model.mu                                            # shape = (n_classes,dim)
    m = np.empty((n_classes, ))
    offset = np.empty((n_classes, dim))
    for c in range(n_classes):
        if ns_per_class[c] == 0:
            m[c] = 0
        else:
            # ESL中的式18.4
            m[c] = np.power((1/ns_per_class[c] - 1/n_samples), 0.5)
        # ESL 中的式18.4
        offset[c, :] = np.true_divide(mu[c, :] - x_bar[np.newaxis, :], m[c]*(sigma + s0))
        # ESL 中的式18.5
        offset[c, :] = soft_threshold(offset[c, :], _lambda)
        # ESL 中的式18.7
        mu[c, :] = x_bar + m[c]*(sigma+s0)*offset[c, :]

    model.mu = mu
    model.sigma_pooled_diag = np.power(sigma, 2)
    model.shrunken_centroids = offset

    return model
