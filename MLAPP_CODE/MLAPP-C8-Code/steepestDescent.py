def steepestDescent(x0, stepsize, maxiter, fn):
    # 用于存储训练过程中的值
    x1_list = []
    x2_list = []
    y_list = []
    k = 1
    x = x0
    fx, gx, _ = fn(x)
    while True:
        k += 1
        d = -1*gx
        if stepsize is None:
            [] 