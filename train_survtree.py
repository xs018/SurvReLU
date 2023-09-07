import numpy as np
from sksurv.datasets import get_x_y
from sksurv.tree import SurvivalTree

from utils.common_utils import norm_hr
from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.data_utils import load_cox_gaussian, format_dataset_to_df

from sklearn.tree._criterion import Criterion


if __name__ == "__main__":
    train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_cox_gaussian("datasets/gaussian_survival_data_2d.h5")
    
    train_data_df = format_dataset_to_df(train_data, 'T', 'E')
    X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)

    val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
    X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)

    viz_df = format_dataset_to_df(viz_data, 'T', 'E')
    X_viz, y_viz = get_x_y(viz_df, attr_labels=['E', 'T'], pos_label=1)

    random_state = 20
    max_depth = 10
    st = SurvivalTree(max_depth=max_depth, min_samples_split=10, min_samples_leaf=15, random_state=random_state)
    st.fit(X_train, y_train)
    print(st.score(X_valid, y_valid))

    hr_pred = norm_hr(st.predict(X_viz))

    hr_true = norm_hr(viz_data['hr'])
    x_0 = viz_data_unnorm['x'][:, 0]
    x_1 = viz_data_unnorm['x'][:, 1]
    plot_risk_model_2(x_0, x_1, hr_true, hr_pred, (12, 5), (-1, 1), 'jet', f'SurvTree_hr_depth{max_depth}.jpg')