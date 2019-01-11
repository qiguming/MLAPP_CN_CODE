import numpy as np
def zero_one_loss_fn(y_hat, y):
    """根据0-1损失返回对比结果
    Input:
    y_hat,预测结果，shape=(n_samples,)
    y,    真实结果, shape=(n_samples,)
    Output:
    false_pred, 预测向量，{0,1}^n_samples, 1代表预测错误，0代表预测正确
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    false_pred = (y_hat.flatten()!= y.flatten())
    return false_pred
