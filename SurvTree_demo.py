import numpy as np
from sksurv.datasets import get_x_y
# from sksurv.tree import SurvivalTree
from baselines.RSF import SurvivalTree

from sklearn.utils import check_random_state

from utils.common_utils import norm_hr
from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.data_utils import load_cox_gaussian, format_dataset_to_df

from lifelines import datasets

#define function to swap columns
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

if __name__ == "__main__":
    train_data, valid_data, test_data, viz_data, viz_data_unnorm = load_cox_gaussian()
    
    train_data_df = format_dataset_to_df(train_data, 'T', 'E')
    train_data_df = swap_columns(train_data_df, 'E', 'T')
    X_train, y_train = train_data_df.iloc[:, :10], train_data_df.iloc[:, 10:]
    
    # X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)

    val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
    val_data_df = swap_columns(val_data_df, 'E', 'T')
    X_valid, y_valid = val_data_df.iloc[:, :10], val_data_df.iloc[:, 10:]
    # X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)

    viz_df = format_dataset_to_df(viz_data, 'T', 'E')
    viz_df = swap_columns(viz_df, 'E', 'T')
    X_viz, y_viz = viz_df.iloc[:, :10], viz_df.iloc[:, 10:]
    # X_viz, y_viz = get_x_y(viz_df, attr_labels=['E', 'T'], pos_label=1)

    random_state = 20
    max_depth = 10
    random_instance = check_random_state(20)

    n_features = X_train.shape[1]
    # print(n_features)

    f_idxs = list(range(n_features))

    # print(f_idxs)
    timeline = y_train.iloc[:, 1].sort_values().unique()
    st = SurvivalTree(X_train, y_train, f_idxs, n_features, random_instance, timeline)

    # st = SurvivalTree(max_depth=max_depth, min_samples_split=10, min_samples_leaf=15, random_state=random_state)
    # st.fit(X_train, y_train)
    # print(st.score(X_valid, y_valid))

    # hr_pred = norm_hr(st.predict(X_viz))
    # hr_true = norm_hr(viz_data['hr'])
    # x_0 = viz_data_unnorm['x'][:, 0]
    # x_1 = viz_data_unnorm['x'][:, 1]
    # plot_risk_model_2(x_0, x_1, hr_true, hr_pred, (12, 5), (-1, 1), 'jet', f'SurvTree_hr_depth{max_depth}.jpg')