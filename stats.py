import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

## Metabric
## fold [0.6680, 0.6572, 0.6469, 0.6728, 0.6521]

# arr = np.array([0.6680, 0.6572, 0.6469, 0.6728, 0.6521])

# print(arr.mean(), arr.std())

## depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
# depth_performance = np.array([0.4921, 0.4987])

# depth_performance = np.load('viz/depth_ablation.npy')
# print(depth_performance.max())

# plt.plot(depth_performance)
# plt.show()

# depth = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
# depth_performance = np.array([0.5940, 0.4926, 0.6066, 0.5976, 0.6266, 0.6441,
#                               0.6542, 0.6477, 0.6534, 0.6462, 0.6590,
#                               0.6545, 0.6566, 0.6570, 0.6564, 0.6595,
#                               0.6576, 0.6561, 0.6579, 0.6576, 0.6547])

# plt.plot(depth, depth_performance, marker = 'o', linewidth=2, ms = 8)
# plt.plot([30], [0.6595], marker = '*', linewidth=2, ms = 18, mec = 'r', mfc = 'r')
# yy = np.linspace(depth_performance.min(), depth_performance.max(), 100)
# xx = 30 * np.ones((100, ))
# plt.plot(xx, yy, '--', c='red')

# plt.xlabel("Depth", fontsize=12)
# plt.ylabel("C-index", fontsize=12)
# plt.xticks(depth)

# patch1 = mpatches.Patch(color='red', label='Optimal Depth')
# plt.legend(handles=[patch1], loc="upper left", ncol=1, fontsize=8)

# plt.tight_layout()
# plt.savefig("viz/ablation_depth.jpg", dpi=500)

l1_regularization = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
reg_performance = np.array([0.5317, 0.5305, 0.5733, 0.6484, 0.65947164, 0.65951034, 0.659498])
sparisty = np.array([0.9999, 0.9996, 0.9986, 0.659864, 0.4966, 0.4911, 0.4898])

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()

ax.plot(l1_regularization, reg_performance, marker = 'o', linewidth=2, ms = 8, c='C0')
ax.plot([1e-4], [0.65947164], marker = '*', linewidth=2, ms = 18, mec = 'r', mfc = 'r')
ax.set_xscale('log')

yy = np.linspace(reg_performance.min(), reg_performance.max(), 100)
xx = 1e-4 * np.ones((100, ))
ax.plot(xx, yy, '--', c='red')

ax2.plot(l1_regularization, sparisty, marker = 'o', linewidth=2, ms = 8, c='C1')

ax.set_xlabel("Regularization Strangth($\lambda$) [Log-sacle]", fontsize=12)
ax.set_ylabel("C-index", fontsize=12)
ax2.set_ylabel("Sparisty", fontsize=12)
ax.set_xticks(l1_regularization)

# plt.xlabel("Regularization Strangth($\lambda$) [Log-sacle]", fontsize=12)
# plt.ylabel("C-index", fontsize=12)
# plt.xticks(l1_regularization)
patch1 = mpatches.Patch(color='red', label='Optimal $\lambda$')
plt.legend(handles=[patch1], loc="upper center", ncol=1, fontsize=8)

plt.tight_layout()
# plt.show()
plt.savefig("viz/ablation_reg.jpg", dpi=500)
