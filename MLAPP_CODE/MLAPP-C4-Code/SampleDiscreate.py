import numpy as np
def sample_discreate(prob, n_samples):
    """根据类先验分布对标签值进行采样
    M = sample_discreate(prob, n_samples)
    Input:
    prob:         类先验分布        shape=(n_classes,)
    n_samples:    需要采样的数量    shape = (n_samples,)
    Output:
    M:            采样得到的样本类别  shape = (n_samples,)
    例子：
    sample_discreate([0.8,0.2],n_samples)
    从类别[0,1]中采样产生n_samples个样本
    其中采样得到0的概率为0.8，得到1的概率为0.2.
    """
    np.random.seed(1)                     # 使每一次生成的随机数一样
    n = prob.size                         # 类别的数量
    R = np.random.rand(n_samples)        # 生成服从均匀分布的随机数
    M = np.zeros(n_samples)               # 初始化最终结果
    cumprob = np.cumsum(prob)             # 累积概率分布

    if n < n_samples:                     # 如果采样的样本数量大于类别数量   
        for i in range(n-1):
            M = M + np.array(R > cumprob[i])
    else:                                 # 如果采样的样本数量小于类别数量
        cumprob2 = cumprob[:-1]
        for i in range(n_samples):
            M[i] = np.sum(R[i] > cumprob2)
    return M

#  进行相关测试
if __name__ == '__main__':
    print(sample_discreate(np.array([0.8,0.2]),10))
