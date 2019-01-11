import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline        # 在jupyter notebook中使用
plt.rcParams['figure.figsize'] = (15, 5)   # 设置绘图的尺寸

class naiveBayes():
    """定义naiveBayes分类器类"""
    def __init__(self):
        self.theta = None
        self.c_prior = None
    # 定义拟合方法
    def fit(self, X_train, y_train, pseudoCount=1):
        C = np.unique(y_train)    # 获取存在的类别
        N_train, D = X_train.shape   # 获取训练样本的数量和特征的数量
        theta = np.empty((len(C), D))        # 初始化参数矩阵
        c_prior = np.empty(len(C))           # 初始化类先验分布
        for c in C:                          # 由mat导入的数据中类别的编号从1开始
            XX = X_train[y_train.flatten() == c]   # 获取所有属于类别c的样本
            Non = np.sum(XX, 0)              # 统计每个特征取值为1的数量
            Noff = XX.shape[0] - Non         # 统计每个特征取值为0的数量
            theta[c-1, :] = (Non + pseudoCount)/(Non + Noff + 2*pseudoCount)
            c_prior[c-1] = XX.shape[0]/N_train
            self.theta = theta
            self.c_prior = c_prior

    def predict(self, X_test, y_test):
        C = np.unique(y_test)
        Ntest = X_test.shape[0]                          # 获取测试样本的数量
        logPrior = np.log(self.c_prior)
        logPost = np.empty((len(self.c_prior), Ntest))   # 初始化后验预测分布
        logTrue = np.log(self.theta)                     # 计算 logp(xj=1|y=c)
        logTrueNot = np.log(1-self.theta)                # 计算 logp(xj=0|y=c)
        X_test_Not = 1-X_test                            # 取X_test 的反(0->1,1->0)
        for c in C:
            L_true = np.dot(logTrue[c-1,:], X_test.T)     
            L_false = np.dot(logTrueNot[c-1,:], X_test_Not.T)
            logPost[c-1,:] = L_true + L_false + logPrior[c-1]
        y_hat = np.argmax(logPost,0) + 1                   # 我们将预测的类别加1，从而实现与原始数据中的类别标签一致
        error_rate = 1- y_hat[y_hat == y_test.flatten()].size/y_hat.size   # 计算误分类率
        print("误分类率为：", error_rate)

data_file = 'XwindowsDocData.mat'
load_data = sio.loadmat(data_file)   # 导入原始数据
X_train = load_data['xtrain']
#print(type(X_train), X_train.shape)
y_train = load_data['ytrain']
#print(type(y_train), y_train.shape)

myNBC = naiveBayes()                # 对模型进行训练，实例化分类器
myNBC.fit(X_train, y_train)

# 绘制图形
ax1 = plt.subplot(121)
ax1.vlines(np.arange(myNBC.theta.shape[-1]), 0, myNBC.theta[0,:], colors='blue')
ax1.set_ylim(0, 1)
ax1.set_title('p(xj=1|y=1)')
ax2 = plt.subplot(122)
ax2.vlines(np.arange(myNBC.theta.shape[-1]), 0, myNBC.theta[1,:], colors='blue')
ax2.set_ylim(0, 1)
ax2.set_title('p(xj=1|y=2)')

# 测试数据
X_test = load_data['xtest'].toarray()
# print(X_test.shape)
y_test = load_data['ytest']
myNBC.predict(X_test, y_test)

# 计算得到MI
C = np.unique(y_train)
N_train, D = X_train.shape
theta_j = np.dot(myNBC.c_prior[np.newaxis, :], myNBC.theta)
I = np.empty((len(C), D))
for c in C:
    I[c-1,:] = myNBC.theta[c-1,:]*myNBC.c_prior[c-1]*np.log(myNBC.theta[c-1,:]/theta_j)+ \
    (1-myNBC.theta[c-1,:])*myNBC.c_prior[c-1]*np.log((1-myNBC.theta[c-1,:])/(1-theta_j))
MI = np.sum(I,0)

# 绘制pandas表格
vocab = load_data['vocab']
cols = ['class1','prob', 'class2', 'prob', 'highest MI', 'MI']
first_col = [element[0] for element in vocab[np.argsort(-myNBC.theta[0,:])[0:5]].flatten()]
second_col = -np.sort(-myNBC.theta[0,:])[0:5]
third_col = [element[0] for element in vocab[np.argsort(-myNBC.theta[1,:])[0:5]].flatten()]
forth_col = -np.sort(-myNBC.theta[1,:])[0:5]
fivth_col = [element[0] for element in vocab[np.argsort(-MI)[0:5]].flatten()]
sixth_col = -np.sort(-MI)[0:5]
df = pd.DataFrame({'class1':first_col,'prob':second_col,'class2':third_col,'prob':forth_col,
                   'highest MI':fivth_col,'MI':sixth_col}
)
df.loc[:,cols]
print(df)

plt.show()