import numpy as np
def grid_predict(axis_range, resolution, predict_fn):
    """基于坐标轴取值范围axis_range对整个坐标系进行网格划分
    Input:
    axis_range:   坐标轴取值范围， list = [min_x0, max_x0, min_x1, max_x1]
    resolution:   每个坐标轴需要分割的点的数量，即分辨率
    predict_fn:   用于预测每一个网格点类别的函数  
    Output:
    X1:   整个网格点的轴0坐标
    X2:   整个网格点的轴1坐标      
    """
    X1range = np.linspace(axis_range[0], axis_range[1], resolution)
    X2range = np.linspace(axis_range[2], axis_range[3], resolution)
    X1, X2 = np.meshgrid(X1range, X2range)      # X1.shape = X2.shape = (len(X2range),len(X1range))
    X = np.dstack((X1, X2))                      # X.shape = (X1.shape[0],X1.shape[1],2)                  
    X = np.reshape(X, (-1, int(len(axis_range)/2))) # X.shape = (n_samples,2)
    # print(X.shape)
    yhat, loglik = predict_fn(X)
    return [X1, X2, yhat, loglik]