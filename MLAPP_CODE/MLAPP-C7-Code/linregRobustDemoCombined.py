import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from scipy.optimize import minimize

x = np.sort(np.random.rand(10))
y = 1 + 2*x + np.random.rand(x.size) - 0.5
 # 添加异常值
x = np.append(x, [0.1,0.5,0.9])
k = -5
y = np.append(y, [k, k ,k])
plt.scatter(x,y,marker='o',color='black')

n = len(x)
Xtrain = np.expand_dims(x, axis=1)
ytrain = np.expand_dims(y, axis=1)

# 使用高斯分布作为似然函数
lr = LinearRegression()
lr.fit(Xtrain,ytrain)
Xtest = np.arange(0,1,0.1)
gaussYhat = lr.predict(np.expand_dims(Xtest, axis=1))
plt.plot(Xtest, gaussYhat, color='r',label="least squares")
plt.scatter(Xtest, gaussYhat, marker='o', linewidths=2, color='', edgecolors='r')

#plt.show()


# 使用拉普拉斯分布作为似然函数
f = np.append(np.zeros(2), np.ones(2*n))
Aeq = np.hstack((np.ones((n,1)), Xtrain, np.eye(n), -1*np.eye(n)))
beq = ytrain
l = [None]*(2) + [0]*(2*n)
u = [None]*(2+2*n)
lu = zip(l,u)
res = linprog(f, A_eq=Aeq, b_eq=beq, bounds=tuple(lu))
laplace_w = res.x[0:2]
laplaceYhat = laplace_w[1]*Xtest + laplace_w[0]
plt.plot(Xtest, laplaceYhat, color='b',label="laplace")
plt.scatter(Xtest, laplaceYhat, marker='s', linewidths=2, color='', edgecolors='b')



# 使用huber损失作为目标函数,这一块内容目前不够正确，待定
def HuberLoss(w):
    #r = np.dot(np.hstack((np.ones((n,1)),Xtrain)), w) - ytrain
    r = w[0] + w[1]*Xtrain-ytrain
    temp1 = r[np.abs(r)<= 1]
    if len(temp1) == 0: temp1=0
    temp2 = r[np.abs(r) > 1]
    if len(temp2) == 0: temp2=0
    temp3 = np.abs(r) > 1
    if len(temp3) == 0: temp3=0
    f= (1/2)*np.sum(temp1**2) + 1*np.sum(np.abs(temp2)) - 1/2*np.sum(temp3)*(1**2)
    return f
w_int = (lr.intercept_[0], lr.coef_[0][0])
loss = HuberLoss(w_int)
res = minimize(lambda w:HuberLoss(w), w_int)
huber_w = res.x
huberYhat = huber_w[0] + huber_w[1]*Xtest
plt.plot(Xtest, huberYhat, color='green', label='huberloss')
plt.scatter(Xtest, huberYhat, marker='s', color='', edgecolors='green', linewidths=2)

plt.legend()
plt.xlim([min(Xtest),max(Xtest)])
plt.show()







