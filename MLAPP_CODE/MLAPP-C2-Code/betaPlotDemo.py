
# coding: utf-8

# In[11]:


from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize']=(7,5)


# In[3]:


a=np.array([0.1,1.0,2.0,8.0])
b=np.array([0.1,1.0,3.0,4.0])


# In[14]:


for (new_a,new_b) in zip(a,b):
    print(new_a,new_b)
    x = np.linspace(beta.ppf(0.0001,new_a,new_b),
                   beta.ppf(0.9999,new_a,new_b),100)
    plt.plot(x,beta.pdf(x,new_a,new_b),label='a={},b={}'.format(new_a,new_b))
ax = plt.gca()
ax.set_ylim((0,3))
ax.set_xlim((0,1))
ax.legend()

