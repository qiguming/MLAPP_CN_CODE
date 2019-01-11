
# coding: utf-8


import scipy.io as sio    # 用于导入mat文件
import seaborn as sns     # 用于绘制散点图
import scipy.stats as stats   # 用于绘制高斯分布图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set(style="white")
plt.style.use({'figure.figsize':(15, 8)})
#sns.set()

def gaussian2d(mean, cov):
    # 绘制2维高斯分布的轮廓线
    # 输入：mean:期望；cov：协方差矩阵
    # 输出：轮廓线上的坐标
    n = 200        # 绘图需要的点的数量
    eigvals, U = np.linalg.eig(cov)   # 矩阵U表示由特征向量组成的正交矩阵
    D = np.diag(eigvals)  # 由特征值构造出来的对角矩阵D
    k = np.power(stats.chi2.ppf(0.95, 2), 0.5)    # 绘制出95%概率质量覆盖的轮廓线，此处使用了卡方分布的相关知识
    t = np.linspace(0, 2*np.pi, n)
    xy = np.array([np.cos(t), np.sin(t)])
    w = np.dot(np.dot(k*U,np. power(D,0.5)), xy)
    z = w + mean[:,np.newaxis]        # z为%95概率质量的轮廓线上的坐标          
    return z

def kindScatter(x_label, y_label, group_label, ax, data, size=100, legendloc=2):
    """实现对类别数据的绘图，根据其类别进行绘图
    参数： x_label:x坐标的特征名;y_label:y坐标的特征名；group_label:类别所在的特征名
           ax:绘图所基于的坐标轴;data:pandas数据类型，绘图所需的数据      
    """
    kind = np.unique(data[group_label])
    if len(kind) > 7:
        print("类别不允许超过7个")
    else:
        markers = ['o', '*', '+', 'x','s', 'p','h']
        col = ['b','r','g','c','y','m','k']
        for i in range(len(kind)):
            xx = data[x_label][data[group_label] == kind[i]]
            yy = data[y_label][data[group_label] == kind[i]]      
            ax.scatter(xx, yy, marker=markers[i], s=size, c=col[i],
            alpha=0.8, label=kind[i])
            plt.legend(loc=legendloc, frameon=True)


filename = 'heightWeight.mat'
raw_data = sio.loadmat(filename)
raw_data.keys()  # 查看数据的键
print(type(raw_data['heightWeightData']))
# 对数据进行初步了解
data = raw_data['heightWeightData']
print(data[:5,:])       # 打印数据前5行，第1列：1表示男性，2代表女性；第2列：体重；第3列：身高

# 将numpy数组转换为pandas数据结构，以使用seaborn绘图
columns = ['sex', 'height', 'Weight']
df = pd.DataFrame(data=data, columns=columns)
df['sex']=df['sex'].astype('category')    # 将sex这一列转换为类别类数据
df['sex'].cat.categories=['male', 'female']  # 将类别1,2转换为相应的类别
# sns.relplot(x='height', y='Weight', hue='sex', style='sex', data=df)   # 利用seaborn进行绘图
ax1=plt.subplot(121)
kindScatter(x_label='height', y_label='Weight', group_label='sex', ax=ax1, data=df)
ax1.set_xlabel('height')
ax1.set_ylabel('Weight')


# 将属于男性的数据提取出来
df_male = df[df['sex']=='male']
# 将属于女性的数据提取出来
df_female = df[df['sex']=='female']
# 对两组数据进行统计，获取高斯分布的期望和协方差矩阵的MLE
cov = df_male.cov(),df_female.cov()  
mean = df_male.mean(),df_female.mean()

ax2 = plt.subplot(122)
kindScatter(x_label='height', y_label='Weight', group_label='sex', ax=ax2, data=df)
col = ['b','r','g','c','y','m','k']
for i in range(len(cov)):
    z = gaussian2d(mean[i],cov[i])
    ax2.plot(z[0,:],z[1,:],color=col[i])
ax2.set_xlabel('height')
ax2.set_ylabel('Weight')

plt.show()

