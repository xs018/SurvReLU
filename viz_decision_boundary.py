import numpy as np
from sklearn import tree
from sksurv.tree import SurvivalTree
import matplotlib.pyplot as plt

from sksurv.datasets import get_x_y
from utils.data_utils import load_dataset, format_dataset_to_df

plot_step = 0.01
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                    np.arange(y_min, y_max, plot_step))

print(xx.shape, yy.shape, np.c_[xx.ravel(), yy.ravel()].shape)

random_state = 20
dataset = 'simulated_gaussian_2d'
data = load_dataset(dataset)
train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']
train_data_df = format_dataset_to_df(train_data, 'T', 'E')
X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)
val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)
test_data_df = format_dataset_to_df(test_data, 'T', 'E')
X_test, y_test = get_x_y(test_data_df, attr_labels=['E', 'T'], pos_label=1)
viz_df = format_dataset_to_df(viz_data, 'T', 'E')
X_viz, y_viz = get_x_y(viz_df, attr_labels=['E', 'T'], pos_label=1)

model = SurvivalTree(random_state=random_state)
model.fit(X_train, y_train)
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
# ax.contourf(xx, yy, Z, colors=['r'],  alpha=0.2)

text_representation = tree.export_text(model)
print(text_representation)