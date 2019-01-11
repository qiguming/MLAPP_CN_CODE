import numpy as np
def data_window(X):
    """
    Input:
    X:      采样得到的样本点坐标，  shape=(n_samples,2)
    window: 坐标轴范围界限，        list = [min_x0,max_x0,min_x1,max_x1]
    """
    assert(X.shape[1]==2)           # 检查X的形状是否是(n_samples,2)
    minX = np.min(X,0)              # =[minX0,minX1]
    maxX = np.max(X,0)              # =[maxX0,maxX1]
    dx0, dx1 = 0.15*(maxX-minX)     # 在坐标轴两边进行一定的延伸
    window = [minX[0]-dx0, maxX[0]+dx0, minX[1]-dx1, maxX[1]+dx1]
    return window