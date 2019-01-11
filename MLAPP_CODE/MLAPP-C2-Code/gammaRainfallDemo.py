
# coding: utf-8

# In[13]:


from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize']=(7,5)


# In[18]:


a1,a2,a3,b = 1.0,1.5,2.0,1.0
x1=np.linspace(gamma.ppf(0.01,a=a1,scale=b),gamma.ppf(0.99,a=a1,scale=b),1000)
x2=np.linspace(gamma.ppf(0.01,a=a2,scale=b),gamma.ppf(0.99,a=a2,scale=b),1000)
x3=np.linspace(gamma.ppf(0.01,a=a3,scale=b),gamma.ppf(0.99,a=a3,scale=b),1000)

plt.plot(x1,gamma.pdf(x1,a=a1,scale=b),linewidth=2,ls='-',label='a=1,b=1')
plt.plot(x2,gamma.pdf(x1,a=a2,scale=b),linewidth=2,ls='--',label='a=1.5,b=1')
plt.plot(x3,gamma.pdf(x1,a=a3,scale=b),linewidth=2,ls='-.',label='a=2,b=1')
plt.legend()

