#----------------------------导入工具库---------------------------#
"""本程序实现癌症分类"""
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
                os.path.join(data_file, '14cancer.xtrain'), sep='\s+', header=None, encoding='utf-8')
df_ytrain = pd.read_table(\
                os.path.join(data_file,'14cancer.ytrain'), sep='\s+', header=None, encoding='utf-8')
df_xtest = pd.read_table(\
                os.path.join(data_file, '14cancer.xtest'), sep='\s+', header=None, encoding='utf-8')
df_ytest = pd.read_table(\
                os.path.join(data_file, '14cancer.ytest'), sep='\s+', header=None, encoding='utf-8')

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
#--------最近收缩模型
fit_fn_1 = lambda X, y, _lambda: \
                discrim_analysis_fit(X, y, 'shrunkenCentroids', lambdas=_lambda)    # 定义模型训练函数
#--------RDA 模型
fit_fn_2 = lambda X, y, _lambda: \
                discrim_analysis_fit(X, y, 'RDA', lambdas=_lambda)    # 定义模型训练函数
#-------预测函数
predict_fn = lambda model, X: discrim_analysis_predict(model, X)      # 定义模型预测函数

# -------------通过交叉验证得到最近收缩模型的最优参数
#    配置超参数
n_lambdas = 20  
lambdas = np.linspace(0, 8, n_lambdas)
n_folds = 10                           # 交叉验证所需要的fold数目

print('start cross validation for shrunkenCentroids....')
best_model_for_shrunken, best_param_for_shrunken, error_cv_for_shrunken, se_for_shrunken = \
              fit_cv(lambdas, fit_fn_1, predict_fn, zero_one_loss_fn, X_train, y_train, n_folds)
print('*****最近收缩质心模型通过交叉验证得到的最佳参数为{}******'.format(best_param_for_shrunken))
print('cross validation for shrunkenCentroids has finished.')
# 基于最好的最近质心收缩模型，在测试集上进行性能分析
y_hat_index_for_shrunken, _ = predict_fn(best_model_for_shrunken, X_test)
y_hat_for_shrunken = y_test[y_hat_index_for_shrunken]
error_num_for_shrunken = np.sum(zero_one_loss_fn(y_hat_for_shrunken,y_test))
print('最近收缩模型在测试集上预测错误的样本数为{}'.format(error_num_for_shrunken))

print('start cross validation for RDA....')
best_model_for_RDA, best_param_for_RDA, error_cv_for_RDA, se_for_RDA = \
              fit_cv(lambdas, fit_fn_2, predict_fn, zero_one_loss_fn, X_train, y_train, n_folds)
print('*****RDA模型通过交叉验证得到的最佳参数为{}******'.format(best_param_for_RDA))
print('cross validation for shrunkenCentroids has finished.')
# 基于最好的RDA模型，在测试集上进行性能分析
y_hat_index_for_RDA, _ = predict_fn(best_model_for_RDA, X_test)
y_hat_for_RDA = y_test[y_hat_index_for_RDA]
error_num_for_RDA = np.sum(zero_one_loss_fn(y_hat_for_RDA, y_test))
print('RDA模型在测试集上预测错误的样本数为{}'.format(error_num_for_RDA))


