import numpy as np
from tqdm import tqdm
from sksurv.datasets import get_x_y
from sksurv.ensemble  import RandomSurvivalForest
from sksurv.metrics import brier_score, integrated_brier_score

from utils.common_utils import norm_hr
from utils.vis_utils import plot_train_curve, plot_risk_model_2
from utils.data_utils import load_dataset, format_dataset_to_df, load_simulated_data, load_real_data


if __name__ == "__main__":
    dataset = 'metabric'
    data = load_dataset(dataset)

    fold = [3]

    random_state = 4
    np.random.seed(random_state)
    num_bootstrap = 100

    if 'simulated' in dataset:
        vis = True
        train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']
        train_data_df = format_dataset_to_df(train_data, 'T', 'E')
        X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)
        val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
        X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)
        test_data_df = format_dataset_to_df(test_data, 'T', 'E')
        X_test, y_test = get_x_y(test_data_df, attr_labels=['E', 'T'], pos_label=1)
        viz_df = format_dataset_to_df(viz_data, 'T', 'E')
        X_viz, y_viz = get_x_y(viz_df, attr_labels=['E', 'T'], pos_label=1)

        model = RandomSurvivalForest(random_state=random_state)
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

        if vis:
            save_file = f"{dataset}_RSF.npy"
            hr_pred = model.predict(viz_data['x'])
            hr_pred = np.array(hr_pred)
            with open(save_file, 'wb') as f:
                np.save(f, hr_pred)
    else:
        kfold_data = data
        if len(fold) == 5:
            C_index_folds = []
            C_indinces = []
            for fold_idx, fold_data in enumerate(kfold_data):
                train_data, valid_data = fold_data['train'], fold_data['val']
                train_data_df = format_dataset_to_df(train_data, 'T', 'E')
                X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)

                val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
                X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)

                model = RandomSurvivalForest(random_state=random_state)
                model.fit(X_train, y_train)

                C_index_folds.append(model.score(X_valid, y_valid))

            #     indices = np.arange(len(X_valid))
            #     for _ in tqdm(range(num_bootstrap)):
            #         bootstraped_indices = np.random.choice(indices, size=len(X_valid), replace=True)
            #         bootstrapped_X_valid, bootstrapped_y_valid = X_valid.iloc[bootstraped_indices], y_valid[bootstraped_indices]

            #         C_indinces.append(model.score(bootstrapped_X_valid, bootstrapped_y_valid))

            # c_index = float(np.mean(C_index_folds))
            # c_index_lower = np.percentile(C_indinces, 2.5)
            # c_index_upper = np.percentile(C_indinces, 97.5)

            c_index_mean = float(np.mean(C_index_folds))
            c_index_std = float(np.std(C_index_folds))
            print(f"C-Index = {c_index_mean:.4f} (+- {c_index_std:4f})")
        else:
            fold_data = kfold_data[fold[0]]
            train_data, valid_data = fold_data['train'], fold_data['val']
            train_data_df = format_dataset_to_df(train_data, 'T', 'E')
            X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)

            val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
            X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)

            model = RandomSurvivalForest(100, random_state=random_state)
            model.fit(X_train, y_train)

            durations_train = train_data_df['T'].values
            durations_val = val_data_df['T'].values

            c_index = model.score(X_valid, y_valid)
            print(f"C-index:{c_index:.4f}")

            idx = np.logical_and((durations_val > 2.533333), (durations_val < 355.200012))
            durations_val = durations_val[idx] # [2.533333; 355.200012]

            survs = model.predict_survival_function(X_valid)

            time_grid = np.linspace(durations_val.min(), durations_val.max(), 100)[1:-1]

            preds = np.asarray([[fn(t) for t in time_grid] for fn in survs])
            integrated_score = integrated_brier_score(y_train, y_valid, preds, time_grid)

            times, scores = brier_score(y_train, y_valid, preds, time_grid)
            # scores = np.array(scores).squeeze()
            print(integrated_score, scores.shape)

            with open("viz/brier_RSF.npz", 'wb') as f:
                np.savez(f, scores=scores, integrated_score=integrated_score)