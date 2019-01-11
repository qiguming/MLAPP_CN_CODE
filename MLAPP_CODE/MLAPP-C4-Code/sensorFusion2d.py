import numpy as np
from GaussSoftCondition import gauss_soft_condition
import matplotlib.pyplot as plt
from Gaussian2D import gaussian_2d
plt.rcParams["figure.figsize"]=(5, 5)
    
def plot_fun(sigmas):
    y1 = np.array([0, -1])[:, np.newaxis]
    y2 = np.array([1, 0])[:, np.newaxis]
    y = np.vstack((y1, y2))
    class prior_class():
        def __init__(self): pass
    prior_mu = np.array([0, 0])[:, np.newaxis]
    prior_sigma = 1e10*np.eye(2)
    prior=prior_class()
    prior.mu = prior_mu
    prior.sigma = prior_sigma

    A = np.tile(np.eye(2), (2,1))
    class py_class():
        def __init__(self): pass
    py_mu = np.zeros((4, 1))
    py_sigma = np.kron(np.eye(2), sigmas[0])
    index_matrxi = np.array([2,3])[:, np.newaxis]
    py_sigma[index_matrxi, index_matrxi.T]=sigmas[1]
    py = py_class()
    py.mu = py_mu
    py.sigma = py_sigma
    post = gauss_soft_condition(prior, py, A, y)

    z1 = gaussian_2d(y1.flatten(), sigmas[0])
    plt.plot(z1[0,:],z1[1,:],color='red')
    plt.plot(y1.flatten()[0], y1.flatten()[1], 'rx', markersize=5)

    z2 = gaussian_2d(y2.flatten(), sigmas[1])
    plt.plot(z2[0,:], z2[1,:], color='green')
    plt.plot(y2.flatten()[0], y2.flatten()[1], 'gx', markersize=5)

    z3 = gaussian_2d(post.mu.flatten(), post.sigma)
    plt.plot(z3[0, :], z3[1, :], color='black')
    plt.plot(post.mu.flatten()[0], post.mu.flatten()[1], color='black', marker='x', markersize=5)

    plt.show()

sigmas1= (np.tile(0.01*np.eye(2),(2,1))).reshape(2,2,2)
sigmas2_temp = np.array([0.05, 0.01])[:, np.newaxis]
sigmas2 = np.kron(sigmas2_temp, np.eye(2)).reshape(2,2,2)

sigmas3 = 0.01*np.array([[[10, 1],[1, 1]],
                    [[1, 1],[1, 10]]])
for sigmas in [sigmas1, sigmas2, sigmas3]:
    figure = plt.figure()
    plot_fun(sigmas)




