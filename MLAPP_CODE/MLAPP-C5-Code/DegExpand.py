import numpy as np
def deg_expand(x, deg, addOnes=False):
    """对训练样本进行维度拓展
    Input:
    x: 训练样本,shape=(n_samples, dim)
    deg: 拓展阶数, list
    addOnes: 是否需要加1, 默认值为False，默认在第1列加1
    """
    n_samples, dim = x.shape
    n_dim_to_expand = len(deg)
    xx = np.tile(x, (1, n_dim_to_expand))
    for i in range(len(deg)):
        xx[:, dim * i:dim * (i + 1)] = xx[:, dim * i:dim * (i+1)]**(deg[i])
    if addOnes:
        xx = np.insert(xx, 0, values=np.ones(n_samples), axis=1)
    return xx

## 测试代码
if __name__ == "__main__":
    test_array = np.array([[1,2,3],[4,5,6]])
    deg = [1,2,4]
    print(deg_expand(test_array, deg, addOnes=True))