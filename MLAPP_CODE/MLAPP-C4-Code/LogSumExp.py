import numpy as np
def log_sum_exp(a, dim=1):
    """
    返回 log(sum(exp(a),dim))
    a: shape=(n_samples, n_classes)
    """
    y = np.max(a, dim)    # 找出对应维度中最大的值
    # print(max(y))
    a = a - y[:, np.newaxis]
    temp2 = np.sum(np.exp(a), dim)[:, np.newaxis]
    # print(temp2)
    s = y[:, np.newaxis] + np.log(temp2)
    return s