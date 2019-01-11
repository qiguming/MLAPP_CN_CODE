import numpy as np
def soft_threshold(x, data):
    temp1 = (np.abs(x)-data)[np.newaxis, :]
    temp2 = np.zeros((1, len(x)))
    temp = np.append(temp1, temp2, axis=0)
    out = np.sign(x)*np.max(temp, axis=0)
    return out

if __name__ =='__main__':
    x = np.array([1.2,-3.4,5,2])
    data =0
    print(soft_threshold(x,data))