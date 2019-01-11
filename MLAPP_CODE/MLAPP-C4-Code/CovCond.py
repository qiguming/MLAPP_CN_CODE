import numpy as np
from functools import reduce
def cov_cond(c,a):
    """生成条件数为c，第一主元方向向量为a的协方差矩阵
    Input:
    c: 条件数， int
    a: 第一主元方向向量， shape=(dim,1)
    Output:
    sigma: 协方差矩阵 ， shape=(dim,dim)
    preci: 精度矩阵
    """
    dim = a.shape[0]   # 获取维度
    e = np.sort(1/np.linspace(1, c, dim))   # 特征值
    a_div_norm = a/np.linalg.norm(a, axis=0)
    z = np.eye(dim) - 2.0*a_div_norm.dot(a_div_norm.T) # 构建豪斯霍尔德矩阵
    sigma = reduce(np.dot, (z, np.diag(e), z.T))
    preci = reduce(np.dot, (z, np.linalg.inv(np.diag(e)), z.T))
    return sigma, preci

if __name__ == "__main__":
    c = 10; a = np.random.randn(50)[:,np.newaxis]
    sigma, _ = cov_cond(c, a)
    print(np.linalg.cond(sigma, 2))



