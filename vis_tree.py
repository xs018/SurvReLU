import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from lifelines.statistics import logrank_test
plt.rcParams["font.size"] = 20

from batchgenerators.utilities.file_and_folder_operations import *

norm = lambda x, min_v, max_v: (x - min_v) / (max_v - min_v)

dataset = 'metabric'
checklpoint = torch.load(f"viz/best_{dataset}.pth")

weights = checklpoint['state']
valid_data = checklpoint['valid_data']
relu_masks = checklpoint['relu_masks']
model = checklpoint['model']

# print(model.layer[0].weight)

model.load_state_dict(weights)

layer_1 = (weights['layer.0.weight'].data.cpu().numpy().squeeze(), weights['layer.0.bias'].data.cpu().numpy().squeeze())
layer_2 = (weights['layer.1.weight'].data.cpu().numpy().squeeze(), weights['layer.1.bias'].data.cpu().numpy().squeeze())
layer_3 = (weights['layer.2.weight'].data.cpu().numpy().squeeze(), weights['layer.2.bias'].data.cpu().numpy().squeeze())
layer_4 = (weights['layer.3.weight'].data.cpu().numpy().squeeze(), weights['layer.3.bias'].data.cpu().numpy().squeeze())
# layer_5 = (weights['layer.4.weight'].data.cpu().numpy().squeeze(), weights['layer.4.bias'].data.cpu().numpy().squeeze())

layers = [layer_1, layer_2, layer_3, layer_4]

# print(layers[0])
# print(layers[1])
# print(layers[2])
# print(layers[3])
# print(layers[4])

valid_x, valid_t, valid_e = valid_data['x'], valid_data['t'], valid_data['e']

relu_masks_new = []
for l in range(4):
    mask_layer = []
    mask_layer += relu_masks[l].detach().cpu().numpy().squeeze().tolist()
    mask_layer += relu_masks[l+4].detach().cpu().numpy().squeeze().tolist()
    mask_layer += relu_masks[l+8].detach().cpu().numpy().squeeze().tolist()

    relu_masks_new.append(mask_layer)

relu_masks_new = np.array(relu_masks_new)
relu_masks = relu_masks_new.astype(bool)

prev_mask_pos = relu_masks[0, :]
prev_mask_neg = 1 - relu_masks[0, :]
prev_w = None
features = [f'$x_{i}$' for i in range(0, 9) ]
color= [f'C{i}' for i in range(0, 9)]
cur_embed = valid_x[0, :]

index = np.arange(valid_x.shape[0])

node1_pos = np.where(relu_masks[0, :] == 1)[0]
node1_neg = np.where(relu_masks[0, :] == 0)[0]

node1_pos_node2_pos = np.where((relu_masks[0, :] == 1) & (relu_masks[1, :] == 1))[0]
node1_pos_node2_neg = np.where((relu_masks[0, :] == 1) & (relu_masks[1, :] == 0))[0]
node1_neg_node2_pos = np.where((relu_masks[0, :] == 0) & (relu_masks[1, :] == 1))[0]
node1_neg_node2_neg = np.where((relu_masks[0, :] == 0) & (relu_masks[1, :] == 0))[0]

node1_pos_node2_pos_node3_pos = np.where((relu_masks[0, :] == 1) & (relu_masks[1, :] == 1) & (relu_masks[2, :] == 1))[0]
node1_pos_node2_pos_node3_neg = np.where((relu_masks[0, :] == 1) & (relu_masks[1, :] == 1) & (relu_masks[2, :] == 0))[0]
node1_pos_node2_neg_node3_pos = np.where((relu_masks[0, :] == 1) & (relu_masks[1, :] == 0) & (relu_masks[2, :] == 1))[0]
node1_pos_node2_neg_node3_neg = np.where((relu_masks[0, :] == 1) & (relu_masks[1, :] == 0) & (relu_masks[2, :] == 0))[0]
node1_neg_node2_pos_node3_pos = np.where((relu_masks[0, :] == 0) & (relu_masks[1, :] == 1) & (relu_masks[2, :] == 1))[0]
node1_neg_node2_pos_node3_neg = np.where((relu_masks[0, :] == 0) & (relu_masks[1, :] == 1) & (relu_masks[2, :] == 0))[0]
node1_neg_node2_neg_node3_pos = np.where((relu_masks[0, :] == 0) & (relu_masks[1, :] == 0) & (relu_masks[2, :] == 1))[0]
node1_neg_node2_neg_node3_neg = np.where((relu_masks[0, :] == 0) & (relu_masks[1, :] == 0) & (relu_masks[2, :] == 0))[0]

T1, E1 = valid_t[node1_neg_node2_neg_node3_pos], valid_e[node1_neg_node2_neg_node3_pos]
T2, E2 = valid_t[node1_neg_node2_neg_node3_neg], valid_e[node1_neg_node2_neg_node3_neg]
results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
print(results.test_statistic)
print(results.p_value)

### plot layer 2
w1, b1 = layers[0][0], layers[0][1]
w2, b2 = layers[1][0], layers[1][1]
w3, b3 = layers[2][0], layers[2][1]
w4, b4 = layers[3][0], layers[3][1]

### layer 2
layer2_w_pos = w2[:9] + w2[-1] * w1
layer2_w_neg = w2[:9]

# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111)
# ax.bar(features, layer2_w_pos, color=color)
# plt.yticks([])
# plt.tight_layout()
# plt.savefig(f"viz/tree/Layer{2}/w_pos.jpg", dpi=300)

# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111)
# ax.bar(features, layer2_w_neg, color=color)
# plt.yticks([])
# plt.tight_layout()
# plt.savefig(f"viz/tree/Layer{2}/w_neg.jpg", dpi=300)

# T1, E1 = valid_t[node1_pos], valid_e[node1_pos]
# T2, E2 = valid_t[node1_neg], valid_e[node1_neg]
# results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
# print(results.test_statistic)
# print(results.p_value)

### layer 3
layer2_pos_layer3_pos_w = w3[:9] + w3[9] * w1 + w3[10] * layer2_w_pos
layer2_pos_layer3_neg_w = w3[:9] + w3[9] * w1
layer2_neg_layer3_pos_w = w3[:9] + w3[10] * layer2_w_neg
layer2_neg_layer3_neg_w = w3[:9]

### layer 4
layer2_pos_layer3_pos_w_layer4_pos = w4[:9] + w4[9] * w1 + w4[10] * layer2_w_pos + w4[11] * layer2_pos_layer3_pos_w
layer2_pos_layer3_pos_w_layer4_neg = w4[:9] + w4[9] * w1 + w4[10] * layer2_w_pos
layer2_pos_layer3_neg_w_layer4_pos = w4[:9] + w4[9] * w1 + w4[10] * layer2_w_pos + w4[11] * layer2_pos_layer3_neg_w
layer2_pos_layer3_neg_w_layer4_neg = w4[:9] + w4[9] * w1 + w4[10] * layer2_w_pos

layer2_neg_layer3_pos_w_layer4_pos = w4[:9] + w4[10] * layer2_w_pos + w4[11] * layer2_pos_layer3_pos_w
layer2_neg_layer3_pos_w_layer4_neg = w4[:9] + w4[10] * layer2_w_pos
layer2_neg_layer3_neg_w_layer4_pos = w4[:9] + w4[10] * layer2_w_pos + w4[11] * layer2_pos_layer3_neg_w
layer2_neg_layer3_neg_w_layer4_neg = w4[:9] + w4[10] * layer2_w_pos

# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111)
# ax.bar(features, layer2_neg_layer3_neg_w_layer4_neg, color=color)
# plt.yticks([])
# plt.tight_layout()
# plt.savefig(f"viz/tree/Layer{4}/layer2_neg_layer3_neg_w_layer4_neg.jpg", dpi=300)


# for i in range(4):
#     save_dir = f"viz/tree/Layer{i+1}"
#     maybe_mkdir_p(save_dir)
#     num_nodes = 2 ** i
#     print()
#     next_embed = cur_embed @ layers[i][0].reshape(-1, 1) + layers[i][1]
#     print(next_embed)

#     # w = np.abs(layers[i][0])
#     # w = w / w.sum()

#     w = layers[i][0]

#     if i == 0:
#         fig = plt.figure(figsize=(6, 5))
#         ax = fig.add_subplot(111)
#         ax.bar(features, w, color=color)
#         plt.yticks([])
#         plt.tight_layout()
#         plt.savefig(f"{save_dir}/root.jpg", dpi=300)
#     else:
#         w = w[:9] + prev_w * w[-1]
#         w = w / w.sum()

#     prev_w = w

# for i in range(4):
#     T1, E1 = valid_t[relu_masks[i, :] * prev_mask], valid_e[relu_masks[i, :] * prev_mask]
#     T2, E2 = valid_t[1 - relu_masks[i, :] * prev_mask], valid_e[1 - relu_masks[i, :] * prev_mask]
#     results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
#     print(results.p_value)
#     print(results.test_statistic)

# Create a figure and an array of axes.

# new_layers = []
# for layer in layers:
#     w, b = layer[0], layer[1]
#     sum_all = w.sum() + b
#     w = w / sum_all
#     b = b / sum_all
#     new_layers.append((w, b))

# layers = new_layers
