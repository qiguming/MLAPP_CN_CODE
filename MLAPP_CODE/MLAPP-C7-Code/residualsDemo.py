import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
n_samples = 20
xTrainRaw = np.random.randn(n_samples, 1)
Ntrain = len(xTrainRaw)
xTrain = np.hstack((np.ones((Ntrain,1)), xTrainRaw))
wtrue = [1, 1]
sigma = 1
yTrain = wtrue[0] + wtrue[1]*xTrainRaw + sigma*np.random.randn(Ntrain, 1)

X = xTrain
y = yTrain
design_matrix = np.dot(X.T, X)
w = reduce(np.dot, (np.linalg.inv(design_matrix), X.T, y))
yPredTrain = np.dot(xTrain, w)

xTestRaw = np.arange(-3.5,3.5,0.5)[:, np.newaxis]
Ntest = len(xTestRaw)
xTest = np.hstack((np.ones((Ntest,1)), xTestRaw))
yTestOpt = wtrue[0] + wtrue[1]*xTestRaw
yPredTest = np.dot(xTest, w)

plt.plot(xTestRaw.flatten(), yPredTest.flatten(), 'r-', linewidth=2,label='prediction')
plt.plot(xTestRaw.flatten(), yTestOpt.flatten(), 'b:', linewidth=2,label='truth')
plt.plot(xTrainRaw.flatten(), yTrain.flatten(), 'ro', linewidth=2)
plt.plot(xTrainRaw.flatten(), yPredTrain, 'bx', markersize=10, linewidth=2)

for i in range(Ntrain):
    plt.vlines(xTrainRaw.flatten()[i], yPredTrain[i,0], yTrain[i,0], colors='b', linewidth=2)
plt.legend()
plt.show()

