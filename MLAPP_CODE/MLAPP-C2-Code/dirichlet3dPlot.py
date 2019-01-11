
# coding: utf-8

# In[151]:


import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
plt.rcParams['figure.figsize']=(8,8)


# In[152]:


x=np.linspace(0,1,100)
y=np.linspace(0,1,100)
xx,yy=np.meshgrid(x,y)
zz=1-xx-yy
zz
temp_loc=np.dstack((xx,yy,zz))
loc=temp_loc[(zz>0)&(xx>0)&(yy>0)]


# In[167]:


alpha =np.array([0.1,0.1,0.1])
new_pos = loc[0,:]
new_pos
density = [dirichlet.pdf(single_loc,alpha) for single_loc in loc]


# In[163]:


triang = mtri.Triangulation(loc[:,0],loc[:,1])


# In[168]:


plt.tricontourf(triang,density)

