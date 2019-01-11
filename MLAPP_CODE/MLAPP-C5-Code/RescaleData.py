import numpy as np
def rescale_data(x, **keywords):
    """对数据进行rescale
    Input:
    x:  输入的数据， shape=(n_samples, dim)
    keywords:
        min_val : rescale后的最小值
        max_val : rescale后的最大值
        min_x   : 每一维度x的最小值
        range_x : 每一维度的区间长度
    """
    x = x.astype("float")
    min_val = keywords.get("min_val", -1)
    max_val = keywords.get("max_val", 1)
    min_x = keywords.get("min_x", np.min(x, axis=0))
    range_x = keywords.get("range_x", np.max(x, axis=0) - np.min(x, axis=0))

    # rescale到(0:1)
    y = (x - min_x)/range_x
    # rescale到 (0:max_val-min_val)
    y = y * (max_val - min_val)
    # 平移
    y = y + min_val
    return y, min_x, range_x

# 测试代码
if __name__ == "__main__":
    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(rescale_data(x))
