import numpy as np
def canonize_labels(*varargin):
    """
    varargin=(labels,) or varargin=(labels,support)
    labels:  输入的标签列表,            shape=(n_samples,)  # 暂时只支持单标签预测模型
    support: 所有可选标签构成的支撑集，  shape=(n_samples,)
    例子：                  
    假设已知支撑集support=np.range(10,21)，标签为[11,15,17,19]，那么我们可以将11编码为类别1，
    因为我们的支撑集元素是从10开始，相似地将19编码为9。
    所以我们有：
    canonize_labels([10,11,19,20])=[0,1,2,3]                      不指定支撑集情况
    canonize_labels([10,11,19,20],np.linsapce(10,20))=[0,1,9,10]  指定支撑集情况
    """
    labels = varargin[0]
    if len(labels.shape) != 1:
        raise Exception('输入的labels维度必须是1')
    # n_rows ,n_cols = labels.shape
    # labels = np.array(labels).flatten()                # 将2维数组进行展开
    if len(varargin) == 2:
        support = varargin[1]
        labels = np.hstack((labels, support)) # 将所有标签进行合并 
    s, j, canonized = np.unique(labels, return_index = True, return_inverse = True)
    # s: labels中不重复的元素，从小到大排序
    # j: 新的数组s 在labels中第一个出现的索引值
    # canonized: labels 在s 中的索引值

    if len(varargin) == 2:
        # 如果s中的元素与支撑集不相等，则出错，因为如果提供了support，
        # 那么labels必须是support的子集
        s.sort();support.sort()
        if (s!=support).all():
            raise Exception("存在标签值不在支撑集中")
        canonized = canonized[:len(labels)-len(support)]
    
    support = s
    # canonized = np.reshape(canonized,(n_rows,n_cols))
    return canonized, support


##### 以下为该上述函数的测试样例
if __name__ == '__main__':
    labels = np.array([10,11,20])
    support = np.arange(10, 21)
    print(canonize_labels(labels, support))




    

    