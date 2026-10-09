import numpy as np
import matplotlib.pyplot as plt

brier_CPH_Linear = np.load("viz/brier_CPH_Linear.npz")
brier_CPH_MLP_DeepSurv = np.load("viz/brier_CPH_MLP(DeepSurv).npz")
brier_CPH_MLP_CC = np.load("viz/brier_CPH_MLP(CC).npz")
brier_ST = np.load("viz/brier_ST.npz")
brier_RSF = np.load("viz/brier_RSF.npz")
brier_NSOTree = np.load("viz/brier_NSOTree.npz")

time_min, time_max = 5.066666603088379, 351.0
time_grid = np.linspace(time_min, time_max, 100)[1:-1]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

ax.plot(brier_CPH_Linear['scores'], label=f"CPH-Linear\n[IBS={brier_CPH_Linear['integrated_score']:.4f}]")
ax.plot(brier_CPH_MLP_DeepSurv['scores'], label=f"CPH-MLP [DeepSurv)\n(IBS={brier_CPH_MLP_DeepSurv['integrated_score']:.4f}]")
ax.plot(brier_CPH_MLP_CC['scores'], label=f"CPH-MLP (CC)\n[IBS={brier_CPH_MLP_CC['integrated_score']:.4f}]")
ax.plot(brier_ST['scores'], label=f"ST\n[IBS={brier_ST['integrated_score']:.4f}]")
ax.plot(brier_RSF['scores'], label=f"RSF\n[IBS={brier_RSF['integrated_score']:.4f}]")
ax.plot(brier_NSOTree['scores'], label=f"NSOTree\n[IBS={brier_NSOTree['integrated_score']:.4f}]")

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Brier Score", fontsize=12)
ax.set_xticks(np.arange(0, 98, 12), np.arange(0, 401, 50))

yy = np.linspace(0, brier_ST['scores'].max(), 100)
xx = 50 * np.ones((100, ))
ax.plot(xx, yy, '--', linewidth=2)

fig.legend(loc="upper right", ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig("viz/brier.jpg", dpi=500)