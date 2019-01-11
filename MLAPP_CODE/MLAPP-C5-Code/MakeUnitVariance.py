import numpy as np
def make_unit_variance(X, *s):
    """使得X每一列的方差等于1
    Input:
    X: shape=(n_sampels, dim)
    s(optional): 标准差，shape=(dim,1)
    Output:
    X
    s
    """
    if len(s) == 0:
        s = np.std(X, axis=0)
        s[s < np.spacing(1)]= 1  # 防止无穷小值出现
    else:
        s = s[0]
    X = X/s[np.newaxis, :]
    return X, s

if __name__ == "__main__":
    test_X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    std = np.array([1,2,3])
    print(make_unit_variance(test_X))


