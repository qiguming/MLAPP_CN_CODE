
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,5)


# In[39]:


bins=[1,2,3,4,5]
data1=[1,2,3,4]
data2=[1]


# In[62]:


fontdict = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
ax1=plt.subplot(121)
n1=ax1.hist(data1,bins=bins,align='left',rwidth=0.5,density=True)[0]
ax1.set_ylim((0,0.5))
ax1.set_xlabel('(a)',fontdict=fontdict)
for i,j in zip(bins[:len(n1[n1>0])],n1[n1>0]):
    ax1.text(i-0.15,j+0.01,'%s'%j,fontdict=fontdict)
ax2=plt.subplot(122)
n2=ax2.hist(data2,bins=bins,align='left',rwidth=0.5,density=True)[0]
ax2.set_ylim((0,1.2))
ax2.set_xlabel('(b)',fontdict=fontdict)
for i,j in zip(bins[:len(n2[n2>0])],n2[n2>0]):
    ax2.text(i-0.15,j+0.01,'%s'%j,fontdict=fontdict)
ax1.tick_params(labelsize=15)
ax2.tick_params(labelsize=15)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()+ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax1.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax2.xaxis.set_major_locator(plt.MultipleLocator(1.0))

