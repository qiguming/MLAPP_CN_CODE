from Kfold import k_fold
import numpy as np
def cv_estimate(fit_fn, pred_fn, loss_fn, X, y, n_folds, **varargin):
    """
    Input:
    model = fit_fn(X_train, y_train)
    y_hat = pred_fn(X_test)
    L = loss_fn(y_hat, y_test)
    X,         设计矩阵, shape=(n_sampels,dim)
    y,         类标签,   shape=(n_samples,)
    n_folds,   交叉验证需要的fold

    可选参数:
    varargin= {'randomizeorder':randomize_order,'testfolds':test_folds}
    分别表示：是否对原数据进行打乱：{0,1}
             额外指定的test_folds: [np.array_1,np.array_2,...,np.array_folds]
    """
    randomize_order = varargin.get('randomizeorder', False)  # 0 代表不对数据进行打乱
    test_folds = varargin.get('testfolds', [])           # 获取额外指定的测试fold 
    n_samples = X.shape[0]
    if not test_folds:    # 如果未指定测试用的fold
        train_folds, test_folds = k_fold(n_samples, n_folds, randomize_order)
    else:
        all_index = set(range(n_samples))
        train_folds = [np.array(list(all_index-set(single_array))) for single_array in test_folds]
    loss = np.zeros((n_samples, ))
    for f in range(len(train_folds)):
        X_train = X[train_folds[f]]; X_test = X[test_folds[f]]
        y_train = y[train_folds[f]]; y_test = y[test_folds[f]]
        model = fit_fn(X_train, y_train)
        y_hat_index, _ = pred_fn(model, X_test)
        y_hat = model.support[y_hat_index]
        loss[test_folds[f]] = loss_fn(y_hat, y_test)

    mu = np.mean(loss)
    se = np.std(loss)/np.power(n_samples,0.5)

    return mu, se
