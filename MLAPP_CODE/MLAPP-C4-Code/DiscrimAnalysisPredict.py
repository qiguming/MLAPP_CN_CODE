import numpy as np
from LogSumExp import log_sum_exp
from GaussLogProb import gauss_log_prob

def discrim_analysis_predict(model, X_test):
    """使用类条件概率为多维高斯分布的贝叶斯公式
    计算 post[i,c]=p(y=c|x[i,:],parms)
         yhat[i] = argmax_c post[i,c]
    Input:
    model :    已经训练好的DiscrimModel 类
    X_test :   测试样本， shape = (n_samples, dim)
    Output:
    yhat  :    预测的类别的索引号， shape = (n_samples,)
    log_lik  : 每个样本的似然函数值， shape = (n_samples, n_classes)
    """
    n_sampels, _ = X_test.shape
    class_prior = model.class_prior
    n_classes = len(model.class_prior)
    log_lik = np.empty((n_sampels, n_classes))      # 初始化对数似然函数 
    for c in range(n_classes):
        model_type = model.model_type.lower()
        if model_type == 'linear':
            log_lik[:, c] = gauss_log_prob(model.mu[c, :], model.sigma_pooled, X_test)
        elif model_type in ['qda', 'quadratic']:
            log_lik[:, c] = gauss_log_prob(model.mu[c, :], model.sigma[c, :, :], X_test)
        elif model_type == 'diag':
            log_lik[:, c] = gauss_log_prob(model.mu[c, :], model.sigma_diag[c, :], X_test)
        elif model_type == 'shrunkencentroids':
            log_lik[:, c] = gauss_log_prob(model.mu[c, :], model.sigma_pooled_diag, X_test)
        elif model_type == 'rda':
            beta = model.beta[c, :]
            gamma = -(1/2)*np.dot(model[c], beta)
            log_lik[:,c] = X_test.dot(beta[:, np.newaxis]) + gamma
        else:
            raise Exception('无法识别的类型')
    # 对数联合概率 logp(x,y)= log_lik + log_class_prior
    log_joint = log_lik + np.log(class_prior)
    log_post = log_joint - log_sum_exp(log_joint, 1)
    post = np.exp(log_post)      # 后验概率
    y_hat = np.argmax(post, 1)   # 预测的类别的索引值
    return y_hat, log_lik


