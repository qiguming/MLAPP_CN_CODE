import numpy as np
def HuberLoss(w, X, y, k):
    w = np.array([w0, w1])[:, np.newaxis]
    r = np.dot(X, w) - y
    mask = (np.abs(r)<= k)
    f= (1/2)*np.sum(r[mask]**2,axis=0) + k*np.sum(np.abs(r[not mask])) - 1/2*np.sum(np.abs(r) > k)*(k**2)
    return f
