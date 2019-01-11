import numpy as np
import matplotlib.pyplot as plt
# 定义目标函数
def aokiFn(x):
    if len(x.shape) == 1:
        f = 0.5*(x[0]**2-x[1])**2 + 0.5*(x[0]-1)**2
    else:
        f = 0.5*(x[:,0]**2-x[:,1])**2 + 0.5*(x[:,0]-1)**2
    g = [2*x[0]*(x[0]**2-x[1])+x[0]-1, x[1]-x[0]**2]
    H = np.array([[6*x[0]**2 - 2*x[1] + 1, -2*x[0]],
                [-2*x[0], 1]])
    return f,g,H

# 进行优化
stepsize = [None, 0.1, 0.6]
for m in stepsize:
    x0 = [0, 0]


