import matplotlib.pyplot as plt
plt.axes([0.1, 0.1, 7, 7])
plt.xticks(())
plt.yticks(())
plt.text(.6, .6, 'axes([0.1, 0.1, .8, .8])', ha='center', va='center',
size=20, alpha=.5)
plt.axes([2, 2, 2, 2])
plt.xticks(())
plt.yticks(())
plt.text(.5, .5, 'axes([0.2, 0.2, .3, .3])', ha='center', va='center',
size=16, alpha=.5)
plt.show()