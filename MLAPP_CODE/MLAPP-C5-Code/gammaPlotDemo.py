"""绘制图5.1"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
a_list = np.array([1,1.5,2])
b = 1
b_list = b*np.ones(len(a_list))
styles = ['k:','b--','r-']

for i in range(len(a_list)):
    a = a_list[i]; b = b_list[i]
    x_s = np.linspace(0.1, 7, 40)
    p_s = stats.gamma.pdf(x_s, a=a, loc=0, scale=b)
    plt.plot(x_s, p_s, styles[i], label="a={0},b={1}".format(a,b))
plt.legend()
plt.show()
