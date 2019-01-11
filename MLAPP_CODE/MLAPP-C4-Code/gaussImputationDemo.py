"""实现缺失数据的重构"""
from scipy import stats
import numpy as np
from randpd import rand_pd  # 生成随机正定矩阵
from GaussImpute import gauss_impute
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,12)
DIM = 20    # 数据的维度
N_SAMPLES = 5  # 采集的样本数量
MISS_PRO = 0.5  # 缺失的概率

class model():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
MU = np.random.randn(DIM)
SIGM = rand_pd(DIM)
MODEL = model(MU, SIGM)

x_full = stats.multivariate_normal.rvs(MU, SIGM, N_SAMPLES)
miss_index = np.random.randn(N_SAMPLES, DIM) < MISS_PRO

x_miss = x_full.copy()
x_miss[miss_index] = np.nan

# 数据重构
x_impute, x_impue_var = gauss_impute(MODEL, x_miss)

# 图形绘制
x_miss0 = x_miss.copy(); x_miss0[np.isnan(x_miss)] = 0

n_row = 3; n_column = 5
figure = plt.figure()
for i in range(n_row):
    ax1 = plt.subplot(n_row, n_column, n_column*i+1)
    v_index = np.where(~ np.isnan(x_miss[i, :]))[0]
    ax1.stem(v_index, x_miss[i, v_index])
    if i == 0: plt.title('observed')
    ax1.set_ylim([-5, 5])

    ax2 = plt.subplot(n_row, n_column, n_column*i+2)
    ax2.stem(x_impute[i, :])
    if i == 0:plt.title('imputed')
    ax2.set_ylim([-5, 5])

    ax3 = plt.subplot(n_row, n_column, n_column*i+3)
    ax3.stem(x_full[i, :])
    if i == 0:plt.title('truth')
    ax3.set_ylim([-5, 5])

    ax4 = plt.subplot(n_row, n_column, n_column*i+4)
    ax4.stem(x_full[i, :]-x_impute[i, :])
    if i == 0:plt.title('error')
    ax4.set_ylim([-2, 2])

    ax5 = plt.subplot(n_row, n_column, n_column*i+5)
    upper_y = x_impute[i, :] + 2*np.power(x_impue_var[i, :],0.5)
    lower_y = x_impute[i, :] - 2*np.power(x_impue_var[i, :],0.5)
    ax5.plot(np.arange(DIM), x_impute[i, :], 'bo-', linewidth=0.5, markersize=0.5)
    ax5.fill_between(np.arange(DIM), lower_y, upper_y, color='black')
    if i == 0:plt.title('uncertainty')
    ax5.set_ylim([-4, 4])
plt.show()

