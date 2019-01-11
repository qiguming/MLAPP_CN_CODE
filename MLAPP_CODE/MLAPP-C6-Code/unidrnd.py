import numpy as np
def unid_rnd(n_states, *varargin):
    if len(varargin) == 0:   # 如果未指定样本数目
        size = 1
    else:
        size = varargin[0]   # 获取样本的数量
    r = np.trunc(n_states * np.random.rand(size))
    return r.astype("int")