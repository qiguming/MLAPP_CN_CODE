import numpy as np
def k_fold(n_samples, n_folds, randomize = False):
    """
    计算K-fold交叉验证产生的样本索引值
    Input:
    n_samples, 样本的数量
    n_folds,   分包的数量, 如果n_folds=n_samples,对应留一法交叉验证
    if randomize = True(default,False),  首先对所有样本的索引值打乱，这对于那些样本按照一定
    顺序排列的情况特别适用，比如所有的正样本都在前，负样本都在后。

    例子：
    k_fold(100, 3)

    Output:

    testfolds[0]=np.arange(33),    trainfolds[0]=np.arange(33,100)
    testfolds[1]=np.arange(33,66), trainfolds[1]=np.array(set(np.arange(N))-set(testfolds[1]))
    testfolds[2]=np.arange(66,100),trainfolds[2]=np.arange(66)
    """
    if randomize:           # 如果randomize != 0，对原样本索引值进行打乱
        np.random.seed(0)   # 确保每次打乱的顺序是一样的
        perm = np.random.permutation(n_samples)   # 打乱后的索引值
    else:
        perm = np.arange(n_samples)

    index = 0
    number_in_test = n_samples//n_folds # 每一个test中样本的数目
    test_folds = []      # 初始化测试集索引值列表
    train_folds = []     # 初始化训练集索引值列表

    for i in np.arange(n_folds):
        low_index = index               # 索引的下限值
        if i == n_folds-1:              # 当 当前fold为最后一个fold时
            high_index = n_samples      # 测试集索引的上限值为所有样本的长度，即一直取到最后一个元素
        else:
            high_index = low_index + number_in_test
        test_folds.append(range(low_index, high_index))
        train_folds.append(list(set(range(n_samples))-set(test_folds[-1])))
        test_folds[i] = perm[test_folds[-1]]
        # print(len(test_folds[i]))
        train_folds[i] = perm[train_folds[-1]]
        index = index + number_in_test
    return train_folds, test_folds

if __name__  == '__main__':
    print(k_fold(100, 3, 1))