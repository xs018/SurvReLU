import numpy as np
from tqdm import tqdm
from sksurv.datasets import get_x_y
from sksurv.linear_model import CoxPHSurvivalAnalysis

from utils.common_utils import norm_hr
from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.data_utils import load_dataset, format_dataset_to_df


if __name__ == "__main__":
    dataset = 'support'
    data = load_dataset(dataset)

    random_state = 0
    np.random.seed(random_state)
    num_bootstrap = 100

    if 'simulated' in dataset:
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']
        train_data_df = format_dataset_to_df(train_data, 'T', 'E')
        X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)
        val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
        X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)
        test_data_df = format_dataset_to_df(test_data, 'T', 'E')
        X_test, y_test = get_x_y(test_data_df, attr_labels=['E', 'T'], pos_label=1)
        viz_df = format_dataset_to_df(viz_data, 'T', 'E')
        X_viz, y_viz = get_x_y(viz_df, attr_labels=['E', 'T'], pos_label=1)

        model = CoxPHSurvivalAnalysis()
        model.fit(X_train, y_train)

        C_indinces = []
        indices = np.arange(len(X_test))
        for _ in tqdm(range(num_bootstrap)):
            bootstraped_indices = np.random.choice(indices, size=len(X_test), replace=True)
            bootstrapped_X_test, bootstrapped_y_test = X_test.iloc[bootstraped_indices], y_test[bootstraped_indices]
            
            C_indinces.append(model.score(bootstrapped_X_test, bootstrapped_y_test))

        c_index_lower = np.percentile(C_indinces, 2.5)
        c_index_upper = np.percentile(C_indinces, 97.5)

        c_index = model.score(X_test, y_test)

        print(f"C-Index (95% CI) ={c_index:.4f}({c_index_lower:.4f}, {c_index_upper:.4f})")
        # print(f"Validation C-Index={model.score(X_valid, y_valid)}" )
        # print(f"Test C-Index={model.score(X_test, y_test)}" )
    else:
        kfold_data = data
        C_index_folds = []
        # c_index_lowers = []
        # c_index_uppers = []
        C_indinces = []
        for fold_idx, fold_data in enumerate(kfold_data):
            train_data, valid_data = fold_data['train'], fold_data['val']
            train_data_df = format_dataset_to_df(train_data, 'T', 'E')
            X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)

            val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
            X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)

            model = CoxPHSurvivalAnalysis()
            model.fit(X_train, y_train)
            
            C_index_folds.append(model.score(X_valid, y_valid))

            
            indices = np.arange(len(X_valid))
            for _ in tqdm(range(num_bootstrap)):
                bootstraped_indices = np.random.choice(indices, size=len(X_valid), replace=True)
                bootstrapped_X_valid, bootstrapped_y_valid = X_valid.iloc[bootstraped_indices], y_valid[bootstraped_indices]
                
                C_indinces.append(model.score(bootstrapped_X_valid, bootstrapped_y_valid))

        c_index_lower = np.percentile(C_indinces, 2.5)
        c_index_upper = np.percentile(C_indinces, 97.5)
            # c_index_lowers.append(c_index_lower)
            # c_index_uppers.append(c_index_upper)

        c_index = float(np.mean(C_index_folds))
        # c_index_lower = float(np.mean(c_index_lowers))
        # c_index_lower = float(np.mean(c_index_uppers))

        print(f"C-Index (95% CI) = {c_index:.4f} ({c_index_lower:.3f}, {c_index_upper:.3f})")