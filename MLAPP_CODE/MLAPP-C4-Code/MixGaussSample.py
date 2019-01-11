import numpy as np
from scipy import stats
from SampleDiscreate import sample_discreate
def mix_gauss_sample(mu, sigma, mix_weight, n_samples):
    """对服从混合高斯分布的样本进行采样。首先根据类先验分布对每个样本的类别y进行采样，
    再根据采样得到的类别对样本X进行采样。
    [X, y]=mix_gauss_sample(mu, sigma, mix_weight, n_samples)
    Input:
    mu:             概率分布的期望        shape=(n_classes,dim)
    Sigma:          概率分布的协方差矩阵   shape=(n_classes,dim,dim)
    mix_weight:     类的先验分布          shape=(n_classes,)
    n_samples :     采样的数目
    Output:
    X :             采样得到的样本        shape=(n_samples,dim)
    y :             采样得到的标签值      shape=(n_samples,)
    """
    temp_y = sample_discreate(mix_weight, n_samples)   # 根据类先验分布，对标签y值进行采样
    # print(temp_y,type(temp_y),temp_y.dtype)
    y = temp_y.astype('int16')                        # 将y值转化为整型，因为后面需要将其作为索引值
    dim = mu.shape[1]                                   # 获取维度的数量
    X = np.empty((n_samples, dim))                      # 初始化样本
    for j in range(n_samples):
        X[j,:] = stats.multivariate_normal.rvs(mean=mu[y[j], :], cov=sigma[y[j], :, :], size=1, random_state=None)[np.newaxis, :]
        # print(X[j,:],y[j])
    return X, y