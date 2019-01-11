import numpy as np
def center_cols(X, *mu):
    """对样本进行中心化或按照指定的期望值mu进行中心化
    Input:
    X:  shape=(n_samples, dim)
    mu(optional): 给定的期望值"""
    if len(mu) == 0:  # 未指定期望值
        mu = np.mean(X, axis=0)
    else:
        mu = mu[0]
    X = X - mu[np.newaxis, :]
    return X, mu

if __name__ == "__main__":
    test_X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    mu = np.array([1,2,3])
    print(center_cols(test_X, mu))
