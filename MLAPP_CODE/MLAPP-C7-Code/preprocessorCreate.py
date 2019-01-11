import numpy as np
"""定义训练集的预处理器"""
class standardizeX:
    """对数据进行标准化处理"""
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std
    def preprocess(self, X):
        if self.mu is None:
            self.mu = np.mean(X, axis=0)
        if self.std is None:
            self.std = np.std(X, axis=0)
        X = X - self.mu
        X /= self.std
        return X

class rescaleX:
    def __init__(self, lb=-1, ub=1, xmin=None, xrange=None):
        self.lb = lb
        self.ub = ub
        self.xmin = xmin
        self.xrange= xrange
    def preprocess(self, X):
        if self.xmin is None:
            self.xmin = np.min(X, axis=0)
        if self.xrange is None:
            self.xrange = np.max(X, axis=0) - np.min(X, axis=0)
        X = (X - self.xmin)/self.xrange
        X *= (self.ub - self.lb)
        X += self.lb
        return X

class polyX:
    def __init__(self, degree, flag=False):
        """
        Input: degree, 需要扩展到的维度
        """
        self.degree = degree
    def preprocess(self, X):
        dim = X.shape[1]
        assert(self.degree > 0)
        X = np.tile(X, [1, self.degree])
        degree_list = np.arange(1,self.degree+1,1,dtype=int)
        for i, single_degree in enumerate(degree_list):
            X[:,i*dim : (i+1)*dim] = X[:,i*dim : (i+1)*dim]**single_degree
        return X

class addOne():
    """进行加1处理，注意加1处理应该在所有其他预处理之后"""
    def __init__(self):
        pass
    def preprocess(self, X):
        """
        Input: X, 训练集
        """
        return np.insert(X, 0, 1,axis=1)

        
        
    

    



