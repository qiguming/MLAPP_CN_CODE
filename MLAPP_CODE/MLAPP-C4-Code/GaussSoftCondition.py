import numpy as np
from functools import reduce
def gauss_soft_condition(pmu, py, A, y):
    s_y_inv = np.linalg.inv(py.sigma)
    s_mu_inv = np.linalg.inv(pmu.sigma)
    class post_dis():
        def __init__(self): pass
    post = post_dis()
    post.sigma = np.linalg.inv(s_mu_inv + reduce(np.dot,(A.T, s_y_inv, A)))
    temp = reduce(np.dot, (A.T, s_y_inv, (y-py.mu))) + s_mu_inv.dot(pmu.mu)
    post.mu = np.dot(post.sigma, temp)
    return post