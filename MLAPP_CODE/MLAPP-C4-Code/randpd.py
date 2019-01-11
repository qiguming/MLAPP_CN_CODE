import numpy as np
def rand_pd(dim):
    """
    随机生成一个正定矩阵，维度为dim
    Input:
       dim: 矩阵的维度
    Output:
       M: 正定矩阵
    """
    A = np.random.randn(dim)[:, np.newaxis] # shape=(dim, 1)
    M = A.dot(A.T)
    eig_vals, _ = np.linalg.eig(M)
    while not np.all(eig_vals > 0):
        M += np.diag(0.001*np.ones(dim))
        eig_vals, _ = np.linalg.eig(M)
    return M

#  测试代码
if __name__ == "__main__":
    test_M = rand_pd(4)
    eigs, _ = np.linalg.eig(test_M)
    print(np.all(eigs > 0))

    


