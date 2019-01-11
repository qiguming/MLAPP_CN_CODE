import numpy as np
from CvEstimate import cv_estimate     #   对每个模型进行交叉验证的函数
def fit_cv(params, fit_fn, predict_fn, loss_fn, X, y, n_folds, **varargin):
    """
    训练具有不同复杂度的模型，并通过交叉验证的方法选择最好的模型
    Input:
    params: 一个矩阵,每一行代表一组可调参数(即一个模型)，比如：
            params = [[ lambda1[0],lambda2[0],...,lambdaN[0] ],
                      [ lambda1[1],lambda2[1],...,lambdaN[1] ] ]
    model = fit_fn(X_train, y_train , params[i,:])
    y_hat = predict_fn(model, X_test)
    L = loss_fn(y_hat, y_test)
    X, 设计矩阵, shape=(n_samples, dim)
    y, 样本标签, shape=(n_samples,)
    n_folds, CV_fold数量

    可选参数：
    varargin = {'testfolds':test_folds,'randomizeorder':randomize_order}
    分别表示：
        额外指定的测试分包, [np.array_1,np.array_2,...,np.array_folds], 其中np,array_1表示分包1中用于测试样本的索引值
        是否对提供的样本进行随机打乱

    Output:
    model:       通过交叉验证方法得到的最好的模型
    best_param:  最好模型对应的参数
    mu[i]:       不同模型对应的损失期望值
    se[i]:       不同模型的损失的标准差
    """
    test_folds = varargin.get('testfolds', [])
    randomize_order = varargin.get('randomizeorder', False)
    if len(params.shape) == 1 and len(params) > 1:  # 如果输入的params是1维数组，则可能只是1个参数的不同取值
        params = params[:, np.newaxis]
    n_models = params.shape[0]                      # 获取模型的数量，即参数params的行数
    if n_models == 1:                               # 如果模型的数量为1
        model = fit_fn(X, y, params[0, :])
        best_param = params[0, :]
        mu, se = cv_estimate(lambda X, y: fit_fn(X, y, best_param), predict_fn, loss_fn, X, y, \
                             n_folds, testfolds=test_folds, randomizeorder=randomize_order)
        return model, best_param, mu, se
    mu = np.zeros((n_models, ))                    # 初始化不同模型的损失期望值
    se = np.zeros((n_models, ))                    # 初始化不同模型的损失标准差期望值
    for m in range(n_models):
        print('model {0}/{1} start....'.format(m+1, n_models))
        param = params[m]
        mu[m], se[m] = cv_estimate(lambda X, y: fit_fn(X, y, param), predict_fn, loss_fn, \
                        X, y, n_folds, testfolds=test_folds)
        print('model {0}/{1} finished,the error_rate is {2}'.format(m+1, n_models, mu[m]))
    best_model_index = np.argmin(mu)
    best_param = params[best_model_index]
    model = fit_fn(X, y, best_param)
    return model, best_param, mu, se


