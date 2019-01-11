import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

n=20
x = np.linspace(-5,5,n)[:,np.newaxis]
wtrue = [1,1]
sigma = 1
y = wtrue[0] + wtrue[1]*x + sigma*np.random.randn(n,1)

X = np.hstack((np.ones((n,1)), x))
design_matrix = np.dot(X.T, X)
w = reduce(np.dot, (np.linalg.inv(design_matrix), X.T, y))

v = np.arange(-5,5,0.5)
nv = len(v)

[w0, w1] = np.meshgrid(v,v)
w01 = np.reshape(np.dstack((w0, w1)),(-1,2))

yPred = np.dot(X, w01.T)
res = yPred -y
SR = np.sum(res**2, axis=0)
SR = SR.reshape((nv, nv))

plt.contour(w0, w1 ,SR)
plt.plot(w[0],w[1],'rx', markersize=14, linewidth=3)
ax =plt.gca()
plt.show()