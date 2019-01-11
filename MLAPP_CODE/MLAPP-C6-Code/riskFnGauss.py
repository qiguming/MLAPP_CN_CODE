import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=[15,7]

n_samples = [5,20]
color = ['red','black','blue','yellow','green']
lengends = ['mle','median','fixed','postmean1','postmean5']

n_mu = 30
mu_true = np.linspace(-2,2,n_mu)

for i, n_sample in enumerate(n_samples):
    mse1 = np.ones(n_mu)/n_sample
    mse2 = np.pi/(2*n_sample)*np.ones(n_mu)
    mse3 = (mu_true-0)**2
    mse4 = (n_sample*1 + 1*(0-mu_true)**2)/(n_sample+1)**2
    mse5 = (n_sample*1 + 5**2*(0-mu_true)**2)/(n_sample+5)**2
    mse = [mse1, mse2, mse3, mse4, mse5]
    ax = plt.subplot(1,2,i+1)
    for  j, single_mse in enumerate(mse):
        plt.plot(mu_true, mse[j], color=color[j], label=lengends[j], linewidth=3)
    plt.title('n_sample={}'.format(n_sample))
    plt.legend()
    plt.xlim([-2,2])
    plt.ylim([0,0.5])
    print(mse)
plt.show()


