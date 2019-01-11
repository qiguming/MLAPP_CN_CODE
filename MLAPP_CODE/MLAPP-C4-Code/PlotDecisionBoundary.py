import numpy as np
import matplotlib.pyplot as plt
from DataWindow import data_window
from GridPredict import grid_predict
def plot_decision_boundary(X, Y, predict_fn):
    """绘制采样得到的样本点，针对网格化数据点进行类别预测
    Input:
    X:      采样得到的样本点，  shape=(n_samples,2),这里只考虑2维情况
    Y:      采样点对应的标签值，shape=(n_samples,)
    predict_fn:  函数句柄，                 yhat = predict_fn(X_test)
    """
    resolution = 300                            # 网格点的分辨率
    classes = np.unique(Y)                      # 标签种类
    axis_range = data_window(X)                 # 获取坐标轴范围 
    [X1grid, X2grid, yhat, loglik] = grid_predict(axis_range, resolution, predict_fn)  # 进行网格数据的类别预测
    yhatgrid = yhat.reshape(X1grid.shape)
    #  稀疏化的版本，分辨率低一些
    [X1gridSparse, X2gridSparse, yhatSparse, loglikSparse] = grid_predict(axis_range, resolution / 2.5, predict_fn)
    Xsparse = np.dstack((X1gridSparse, X2gridSparse)).reshape((-1, int(len(axis_range)/2)))
    X1sparse=Xsparse[:, 0];X2sparse=Xsparse[:, 1]
    #  设置绘图的基本元素
    c = ['b','r','g','c','y','m']             #  颜色
    markers = [ '*', '+', 'x','s', 'p','h']   #  标记
    for i in classes:
        plt.scatter(X1sparse[yhatSparse == i], X2sparse[yhatSparse == i], s=0.5, color=c[i])  # 绘制所有的网格点，不同的类用不同颜色
        #level = np.arange(i,len(classes))
        plt.contour(X1grid, X2grid, yhatgrid)    # 绘制等高线，即不同类别的分界线
        plt.scatter(X[:,0][Y==i], X[:,1][Y==i], s=30, marker=markers[i], color = c[i])  # 绘制采样得到的点
    plt.xlim(axis_range[0:2])
    plt.ylim(axis_range[2:4])
    return X1gridSparse, X2gridSparse, loglikSparse