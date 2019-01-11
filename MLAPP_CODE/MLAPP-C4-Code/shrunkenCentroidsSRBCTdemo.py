#----------------------------导入工具库---------------------------#
"""本程序实现最近收缩质心分类模型"""
import os
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DropDfNan import drop_df_nan                            # 导入数据清洗函数
from DiscrimAnalysisFit import discrim_analysis_fit          # 导入模型训练函数
from DiscrimAnalysisPredict import discrim_analysis_predict  # 导入模型预测函数
from ZeroOneLossFn import zero_one_loss_fn                   # 导入0-1损失函数
from FitCv import fit_cv                                     # 导入交叉验证函数
plt.rcParams['figure.figsize'] = (10, 8)

#-------------------------------加载数据----------------------------#

data_file = 'dataset'     # 数据所在的目录
df_xtrain = pd.read_table(\
                os.path.join(data_file, 'khan.xtrain'), sep='\s+', header=None, encoding='utf-8')
df_ytrain = pd.read_table(\
                os.path.join(data_file,'khan.ytrain'), sep='\s+', header=None, encoding='utf-8')
df_xtest = pd.read_table(\
                os.path.join(data_file, 'khan.xtest'), sep='\s+', header=None, encoding='utf-8')
df_ytest = pd.read_table(\
                os.path.join(data_file, 'khan.ytest'), sep='\s+', header=None, encoding='utf-8')

#-------------------------------数据预处理----------------------------#

df_xtrain, df_ytrain = drop_df_nan(df_xtrain, df_ytrain)
df_xtest, df_ytest = drop_df_nan(df_xtest, df_ytest)

# 按照机器学习中的常规习惯，我们将数据进行转置，
# 使shape=(n_samples,dim)，并将其转换为numpy类型
X_train = np.array(df_xtrain.T)
X_test = np.array(df_xtest.T)
y_train = np.array(df_ytrain.T).flatten()    # 将所有的类标签转换为1维数组,程序只支持单标签预测
y_test = np.array(df_ytest.T).flatten()

# 查看调整后的数据的形状
print('训练集的尺寸分别为{0}{1},测试集的尺寸分别为{2}{3}'.\
      format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
print('*********数据加载，并预处理成功*********')

#----------------------------- 对模型进行训练----------------------------#

fit_fn = lambda X, y, _lambda: \
                discrim_analysis_fit(X, y, 'shrunkenCentroids', lambdas=_lambda)    # 定义模型训练函数
predict_fn = lambda model, X: discrim_analysis_predict(model, X)                    # 定义模型预测函数

# 配置超参数
n_lambdas = 20  
lambdas = np.linspace(0, 8, n_lambdas)

# 初始化误分类率
error_train_rate = np.empty((n_lambdas, ))
error_test_rate = np.empty((n_lambdas, ))

for i, single_lambda in enumerate(lambdas):
    print('Epoch {0}/{1} start....'.format(i+1, n_lambdas))

    model = fit_fn(X_train, y_train, single_lambda)                    # 对模型进行训练

    y_hat_train_index, _ = predict_fn(model, X_train)                  # 得到预测标签的索引值
    y_hat_test_index, _ = predict_fn(model, X_test)

    y_hat_train = model.support[y_hat_train_index.astype('int64')]     # 得到预测的类标签
    y_hat_test = model.support[y_hat_test_index.astype('int64')]

    error_train_rate[i] = np.sum(zero_one_loss_fn(y_hat_train, y_train))/len(X_train)     # 计算误分类率
    error_test_rate[i] = np.sum(zero_one_loss_fn(y_hat_test, y_test))/len(X_test)

    if single_lambda == 0:
        print('*******使用对角LDA模型时在测试集上的误分类数为{}******'.format(error_test_rate[i]*len(X_test)))

    print('Epoch {0}/{1} finished'.format(i+1, n_lambdas))

#----------------------------- 交叉验证----------------------------#

n_folds = 10                           # 交叉验证所需要的fold数目
X_all = np.vstack((X_train, X_test))   # 根据原文中的处理方式，将所有样本进行合并
y_all = np.hstack((y_train, y_test))
print('start cross validation....')
best_model, best_param, error_cv, se = \
              fit_cv(lambdas, fit_fn, predict_fn, zero_one_loss_fn, X_all, y_all, n_folds)
print('*****通过交叉验证得到的最佳参数为{}******'.format(best_param))
print('cross validation has finished.')
#----------------------------- 图形绘制----------------------------#
new_figure = plt.figure()
plt.plot(lambdas, error_train_rate, 'bo-', markersize=10, linewidth=2, label='training')
plt.plot(lambdas, error_test_rate, 'r*-', markersize=10, linewidth=2, label='testing')
plt.plot(lambdas, error_cv, 'g+-', markersize=10, linewidth=2, label='cv')

plt.legend(loc='upper left')
plt.tick_params(labelsize=23)
plt.xlabel(r'$\lambda$')
plt.ylabel('Misclassification Error')
plt.show()

#----------------------------- 绘制质心----------------------------#
cent_shrunk = best_model.shrunken_centroids        # 收缩质心

model = fit_fn(X_train, y_train, 0)                # 对应于对角LDA模型

cent_unshrunk = model.shrunken_centroids           # 未收缩质心

n_classes, dim = cent_unshrunk.shape
cent_shrunk_boolen = (cent_shrunk != 0)
cent_shrunk_not_zero = np.sum( (np.sum(cent_shrunk_boolen, axis=0)) > 0 )   # 不全部为0的特征有多少个
print('*****在最优模型的情况下，所需要的特征共{}个*****'.format(cent_shrunk_not_zero))

x_labels = ['(a)', '(b)', '(c)', '(d)']
for g in range(n_classes):
    new_figure = plt.figure()
    plt.plot(range(dim), cent_unshrunk[g,:], color=(0.8, 0.8, 0.8))    #  (0.8,0.8,0.8)为灰色的RGB参数
    plt.plot(range(dim), cent_shrunk[g,:], 'b', linewidth=2)
    plt.title('Class{}'.format(model.support[g]))
    plt.xlabel(x_labels[g])
    plt.show()


