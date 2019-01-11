
# coding: utf-8

# In[78]:


import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize']=(12,12)


# In[89]:


x,y = np.mgrid[-4:4:.01,-4:4:.01]


# In[90]:


pos = np.dstack((x,y))


# In[91]:


fig=plt.figure()
rv1 = multivariate_normal([0.5,-0.2],[[2,1.5],[1.5,2]])
ax1=fig.add_subplot(2,2,1)
ax1.contour(x,y,rv1.pdf(pos))
ax1.set_xlim((-4,4))
ax1.set_ylim((-4,4))
ax1.scatter(0.5,-0.2)
ax1.set_title("cov=[[2,1.5],[1.5,2]]")
ax1.set_xlabel('(a)')

rv2 = multivariate_normal([0.5,-0.2],[[1,0],[0,1.8]])
ax2=fig.add_subplot(2,2,2)
ax2.contour(x,y,rv2.pdf(pos))
ax2.set_xlim((-4,4))
ax2.set_ylim((-4,4))
ax2.scatter(0.5,-0.2)
ax2.set_title("cov=[[1,0],[0,1.8]]")
ax2.set_xlabel('(b)')

rv3 = multivariate_normal([0.5,-0.2],[[1.8,0],[0,1.8]])
ax3=fig.add_subplot(2,2,3)
ax3.contour(x,y,rv3.pdf(pos))
ax3.set_xlim((-4,4))
ax3.set_ylim((-4,4))
ax3.scatter(0.5,-0.2)
ax3.set_title("cov=[[1.8,0],[0,1.8]]")
ax3.set_xlabel('(c)')

rv4 = multivariate_normal([0,0],[[1,0],[0,1]])
ax4=fig.add_subplot(2,2,4,projection='3d')
ax4.plot_surface(x,y,rv4.pdf(pos),cmap=plt.cm.hot)
#plt.contour(x,y,rv.pdf(pos))
ax4.set_xlim((-4,4))
ax4.set_ylim((-4,4))
ax4.set_xlabel('(d)')


plt.show()

