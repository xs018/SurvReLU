import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from utils.common_utils import norm_hr
from utils.data_utils import load_dataset

cmap = 'jet'
dataset = 'simulated_gaussian'

def visualize_hr(dataset, base_dir="viz"):
    data = load_dataset(dataset)
    viz_data = data['viz_unnorm']
    x_0 = viz_data['x'][:, 0]
    x_1 = viz_data['x'][:, 1]
    hr_true = norm_hr(viz_data['hr']).reshape(-1, )

    hr_pred_cox_linear = norm_hr(np.load(f"{base_dir}/{dataset}_CPH(Linear).npy")).reshape(-1, )
    hr_pred_cox_deepsurv = norm_hr(np.load(f"{base_dir}/{dataset}_Cox-MLP(DeepSurv).npy")).reshape(-1, )
    hr_pred_cox_cc = norm_hr(np.load(f"{base_dir}/{dataset}_Cox-MLP(CC).npy")).reshape(-1, )
    hr_pred_cox_st = norm_hr(np.load(f"{base_dir}/{dataset}_ST.npy")).reshape(-1, )
    hr_pred_cox_rsf = norm_hr(np.load(f"{base_dir}/{dataset}_RSF.npy")).reshape(-1, )
    hr_pred_cox_nostree = norm_hr(np.load(f"{base_dir}/{dataset}_Cox-MLP_NOSTree.npy")).reshape(-1, )

    r_cox_linear = pearsonr(hr_true, hr_pred_cox_linear)[0]
    r_cox_deepsurv =pearsonr(hr_true, hr_pred_cox_deepsurv)[0]
    r_cox_cc =pearsonr(hr_true, hr_pred_cox_cc)[0]
    r_cox_st =pearsonr(hr_true, hr_pred_cox_st)[0]
    r_cox_rsf =pearsonr(hr_true, hr_pred_cox_rsf)[0]
    r_cox_nostree =pearsonr(hr_true, hr_pred_cox_nostree)[0]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    clim = (-1, 1)
    im = axes[0, 0].scatter(x=x_0, y=x_1, c=hr_true, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)
    im = axes[0, 1].scatter(x=x_0, y=x_1, c=hr_pred_cox_linear, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)
    im = axes[0, 2].scatter(x=x_0, y=x_1, c=hr_pred_cox_deepsurv, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)
    im = axes[0, 3].scatter(x=x_0, y=x_1, c=hr_pred_cox_cc, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)
    im = axes[1, 1].scatter(x=x_0, y=x_1, c=hr_pred_cox_st, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)
    im = axes[1, 2].scatter(x=x_0, y=x_1, c=hr_pred_cox_rsf, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)
    im = axes[1, 3].scatter(x=x_0, y=x_1, c=hr_pred_cox_nostree, marker='.', cmap=cmap, vmin=0, vmax=1)
    im.set_clim(*clim)

    axes[1, 0].set_axis_off()
    axes[0, 4].set_axis_off()
    axes[1, 4].set_axis_off()

    axes[0, 0].set_title(f"(a) True $h(x)$", fontsize=14)
    axes[0, 1].set_title(f"(b) Cox-Linear $\hat{{h}}(x)$\nPearson ${{r}}$ = {r_cox_linear:.3f}", fontsize=14)
    axes[0, 2].set_title(f"(c) Cox-MLP (DeepSurv) $\hat{{h}}(x)$\nPearson ${{r}}$ = {r_cox_deepsurv:.3f}", fontsize=14)
    axes[0, 3].set_title(f"(d) Cox-MLP (CC) $\hat{{h}}(x)$\nPearson ${{r}}$ = {r_cox_cc:.3f}", fontsize=14)
    axes[1, 1].set_title(f"(e) ST $\hat{{h}}(x)$\nPearson ${{r}}$ = {r_cox_st:.3f}", fontsize=14)
    axes[1, 2].set_title(f"(f) RSF $\hat{{h}}(x)$\nPearson ${{r}}$ = {r_cox_rsf:.3f}", fontsize=14)
    axes[1, 3].set_title(f"(g) NSOTree $\hat{{h}}(x)$\nPearson ${{r}}$ = {r_cox_nostree:.3f}", fontsize=14)
    # axes[0, 7].set_title("True $h(x)$")

    for ax in axes.flatten():
        ax.set_xlabel("$x_0$", fontsize=14)
        ax.set_ylabel("$x_1$", fontsize=14)

    plt.tight_layout()
    cb_ax = fig.add_axes([0.8, 0.085, 0.015, 0.82])
    fig.colorbar(im, cax=cb_ax)

    plt.subplots_adjust(hspace=0.38)

    plt.savefig(f"{base_dir}/viz_{dataset}.jpg", dpi=500)


visualize_hr(dataset)

# data = load_dataset(dataset)
# viz_data = data['viz_unnorm']
# linear_hr_true = norm_hr(viz_data['hr']).reshape(-1, )
# linear_hr_pred_cox_linear = norm_hr(np.load("simulated_linear_CPH(Linear).npy")).reshape(-1, )
# linear_hr_pred_cox_deepsurv = norm_hr(np.load("simulated_linear_Cox-MLP(DeepSurv).npy")).reshape(-1, )
# linear_hr_pred_cox_cc = norm_hr(np.load("simulated_linear_Cox-MLP(CC).npy")).reshape(-1, )
# linear_hr_pred_cox_st = norm_hr(np.load("simulated_linear_ST.npy")).reshape(-1, )
# linear_hr_pred_cox_rsf = norm_hr(np.load("simulated_linear_RSF.npy")).reshape(-1, )
# linear_hr_pred_cox_nostree = norm_hr(np.load("simulated_linear_Cox-MLP_NOSTree.npy")).reshape(-1, )

# linear_x_0 = viz_data['x'][:, 0]
# linear_x_1 = viz_data['x'][:, 1]

# data = load_dataset('simulated_gaussian')
# viz_data = data['viz_unnorm']
# gau_hr_true = norm_hr(viz_data['hr']).reshape(-1, )
# gau_hr_pred_cox_linear = norm_hr(np.load("simulated_gaussian_CPH(Linear).npy")).reshape(-1, )
# gau_hr_pred_cox_deepsurv = norm_hr(np.load("simulated_gaussian_Cox-MLP(DeepSurv).npy")).reshape(-1, )
# gau_hr_pred_cox_cc = norm_hr(np.load("simulated_gaussian_Cox-MLP(CC).npy")).reshape(-1, )
# gau_hr_pred_cox_st = norm_hr(np.load("simulated_gaussian_ST.npy")).reshape(-1, )
# gau_hr_pred_cox_rsf = norm_hr(np.load("simulated_gaussian_RSF.npy")).reshape(-1, )
# gau_hr_pred_cox_nostree = norm_hr(np.load("simulated_gaussian_Cox-MLP_NOSTree.npy")).reshape(-1, )

# gau_x_0 = viz_data['x'][:, 0]
# gau_x_1 = viz_data['x'][:, 1]

# # print(mean_absolute_error(linear_hr_true, linear_hr_pred_cox_linear))
# # print(mean_absolute_error(linear_hr_true, linear_hr_pred_cox_deepsurv))
# # print(mean_absolute_error(linear_hr_true, linear_hr_pred_cox_cc))
# # print(mean_absolute_error(linear_hr_true, linear_hr_pred_cox_st))
# # print(mean_absolute_error(linear_hr_true, linear_hr_pred_cox_rsf))
# # print(mean_absolute_error(linear_hr_true, linear_hr_pred_cox_nostree))

# linear_r_cox_linear = pearsonr(linear_hr_true, linear_hr_pred_cox_linear)[0]
# linear_r_cox_deepsurv =pearsonr(linear_hr_true, linear_hr_pred_cox_deepsurv)[0]
# linear_r_cox_cc =pearsonr(linear_hr_true, linear_hr_pred_cox_cc)[0]
# linear_r_cox_st =pearsonr(linear_hr_true, linear_hr_pred_cox_st)[0]
# linear_r_cox_rsf =pearsonr(linear_hr_true, linear_hr_pred_cox_rsf)[0]
# linear_r_cox_nostree =pearsonr(linear_hr_true, linear_hr_pred_cox_nostree)[0]

# # print("====================================")

# # print(mean_absolute_error(gau_hr_true, gau_hr_pred_cox_linear))
# # print(mean_absolute_error(gau_hr_true, gau_hr_pred_cox_deepsurv))
# # print(mean_absolute_error(gau_hr_true, gau_hr_pred_cox_cc))
# # print(mean_absolute_error(gau_hr_true, gau_hr_pred_cox_st))
# # print(mean_absolute_error(gau_hr_true, gau_hr_pred_cox_rsf))
# # print(mean_absolute_error(gau_hr_true, gau_hr_pred_cox_nostree))

# gau_r_cox_linear = pearsonr(gau_hr_true, gau_hr_pred_cox_linear)[0]
# gau_r_cox_linear = pearsonr(gau_hr_true, gau_hr_pred_cox_deepsurv)[0]
# gau_r_cox_linear = pearsonr(gau_hr_true, gau_hr_pred_cox_cc)[0]
# gau_r_cox_linear = pearsonr(gau_hr_true, gau_hr_pred_cox_st)[0]
# gau_r_cox_linear = pearsonr(gau_hr_true, gau_hr_pred_cox_rsf)[0]
# gau_r_cox_linear = pearsonr(gau_hr_true, gau_hr_pred_cox_nostree)[0]

# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# # fig.supxlabel('$x_0$', fontsize=20, y=0.03)
# # fig.supylabel('$x_1$', fontsize=20, x=0.012)

# clim = (-1, 1)
# im = axes[0, 0].scatter(x=linear_x_0, y=linear_x_1, c=linear_hr_true, marker='.', cmap=cmap, vmin=0, vmax=1)
# im.set_clim(*clim)
# im = axes[0, 1].scatter(x=linear_x_0, y=linear_x_1, c=linear_hr_pred_cox_linear, marker='.', cmap=cmap, vmin=0, vmax=1)
# im.set_clim(*clim)
# im = axes[0, 2].scatter(x=linear_x_0, y=linear_x_1, c=linear_hr_pred_cox_deepsurv, marker='.', cmap=cmap, vmin=0, vmax=1)
# im.set_clim(*clim)
# im = axes[0, 3].scatter(x=linear_x_0, y=linear_x_1, c=linear_hr_pred_cox_cc, marker='.', cmap=cmap, vmin=0, vmax=1)
# im.set_clim(*clim)
# im = axes[1, 1].scatter(x=linear_x_0, y=linear_x_1, c=linear_hr_pred_cox_st, marker='.', cmap=cmap, vmin=0, vmax=1)
# im.set_clim(*clim)
# im = axes[1, 2].scatter(x=linear_x_0, y=linear_x_1, c=linear_hr_pred_cox_rsf, marker='.', cmap=cmap, vmin=0, vmax=1)
# im.set_clim(*clim)
# im = axes[1, 3].scatter(x=linear_x_0, y=linear_x_1, c=linear_hr_pred_cox_nostree, marker='.', cmap=cmap, vmin=0, vmax=1)
# im.set_clim(*clim)

# # clim = (-1, 1)
# # im = axes[0, 0].scatter(x=gau_x_0, y=gau_x_1, c=gau_hr_true, marker='.', cmap=cmap, vmin=0, vmax=1)
# # im.set_clim(*clim)
# # im = axes[0, 1].scatter(x=gau_x_0, y=gau_x_1, c=gau_hr_pred_cox_linear, marker='.', cmap=cmap, vmin=0, vmax=1)
# # im.set_clim(*clim)
# # im = axes[0, 2].scatter(x=gau_x_0, y=gau_x_1, c=gau_hr_pred_cox_deepsurv, marker='.', cmap=cmap, vmin=0, vmax=1)
# # im.set_clim(*clim)
# # im = axes[0, 3].scatter(x=gau_x_0, y=gau_x_1, c=gau_hr_pred_cox_cc, marker='.', cmap=cmap, vmin=0, vmax=1)
# # im.set_clim(*clim)
# # im = axes[1, 1].scatter(x=gau_x_0, y=gau_x_1, c=gau_hr_pred_cox_st, marker='.', cmap=cmap, vmin=0, vmax=1)
# # im.set_clim(*clim)
# # im = axes[1, 2].scatter(x=gau_x_0, y=gau_x_1, c=gau_hr_pred_cox_rsf, marker='.', cmap=cmap, vmin=0, vmax=1)
# # im.set_clim(*clim)
# # im = axes[1, 3].scatter(x=gau_x_0, y=gau_x_1, c=gau_hr_pred_cox_nostree, marker='.', cmap=cmap, vmin=0, vmax=1)
# # im.set_clim(*clim)

# axes[1, 0].set_axis_off()
# axes[0, 4].set_axis_off()
# axes[1, 4].set_axis_off()

# axes[0, 0].set_title("(a) True $h(x)$", fontsize=18)
# axes[0, 1].set_title("(b) Cox-Linear $\hat{h}(x)$", fontsize=18)
# axes[0, 2].set_title("(c) Cox-MLP (DeepSurv) $\hat{h}(x)$", fontsize=18)
# axes[0, 3].set_title("(d) Cox-MLP (CC) $\hat{h}(x)$", fontsize=18)
# axes[1, 1].set_title("(e) ST $\hat{h}(x)$", fontsize=18)
# axes[1, 2].set_title("(f) RSF $\hat{h}(x)$", fontsize=18)
# axes[1, 3].set_title("(g) NSOTree $\hat{h}(x)$", fontsize=18)
# # axes[0, 7].set_title("True $h(x)$")

# for ax in axes.flatten():
#     ax.set_xlabel("$x_0$", fontsize=14)
#     ax.set_ylabel("$x_1$", fontsize=14)

# plt.tight_layout()
# cb_ax = fig.add_axes([0.8, 0.1, 0.015, 0.82])
# fig.colorbar(im, cax=cb_ax)

# plt.savefig("viz_linear.jpg", dpi=500)

