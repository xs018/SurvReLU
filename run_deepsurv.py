# Use DeepSurv from the repo
from utils.data_utils import load_datasets

dataset = load_datasets("datasets/support_train_test.h5")

print(dataset['train']['x'].shape)


# from deepsurv_logger import DeepSurvLogger, TensorboardLogger
# import utils
# import viz

# import numpy as np
# import pandas as pd

# import lasagne
# import matplotlib
# import matplotlib.pyplot as plt