
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform


# In[2]:


plt.rcParams['figure.figsize']=(15,5)


# In[29]:


x1=np.linspace(-1,1,100)
y1=uniform.pdf(x1,loc=-1,scale=2)
x2=np.linspace(0,1,100)
y2=np.power(x2,-0.5)/2
rand_x = uniform.rvs(size=10000)
y3=np.power(rand_x,2)


# In[30]:


ax1=plt.subplot(131)
ax1.plot(x1,y1)
ax1.set_xlabel('(a)')
ax2=plt.subplot(132)
ax2.plot(x2,y2)
ax2.set_xlabel('(b)')
ax3=plt.subplot(133)
bins=np.linspace(0,1,100)
result=ax3.hist(y3,bins,align='left',rwidth=0.7,density=True)
ax3.set_xlabel('(c)')

