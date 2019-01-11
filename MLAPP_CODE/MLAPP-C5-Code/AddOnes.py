import numpy as np
def add_ones(x):
    """在设计矩阵中添加1列1"""
    xx = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)
    return xx