
# coding: utf-8

# In[22]:


from sklearn.datasets import load_iris
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
iris_dataset = load_iris()


# In[4]:


print("鸢尾花数据集的键为:\n{}".format(iris_dataset.keys()))


# In[5]:


# 键DESCR为该数据集的描述
print(iris_dataset['DESCR'][:200]+"\n...")


# In[10]:


print("鸢尾花的种类:{}".format(iris_dataset['target_names']))


# In[11]:


print("鸢尾花的特征：\n{}".format(iris_dataset['feature_names']))


# In[12]:


print("鸢尾花数据的类型：{}".format(type(iris_dataset['data'])))


# In[13]:


print("鸢尾花的数据形状：{}".format(iris_dataset['data'].shape))


# In[14]:


# 查看前5行的数据
print("数据集中前5行的数据：\n{}".format(iris_dataset['data'][:5]))


# In[15]:


print("花朵标签的数据类型：{}".format(type(iris_dataset['target'])))


# In[16]:


print("花朵标签的数据形状:{}".format(iris_dataset['target'].shape))


# In[17]:


print("花朵的标签数据：\n{}".format(iris_dataset['target']))
# 其中0,1,2分别代表类'setosa','versicolor','virginica'


# In[23]:


iris_dataframe = pd.DataFrame(iris_dataset['data'],columns=iris_dataset['feature_names'])
grr = pd.scatter_matrix(iris_dataframe,c=iris_dataset['target'],figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8)

