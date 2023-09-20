import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Locally Constant Networks')
    # Task specification
    parser.add_argument('--dataset', type=str, required=False, default="sim_gaussian")

    # parameters for tuning
    parser.add_argument('--depth', type=int, required=False, default=30,
                        help='the depth of the network (tree)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--drop_type', type=str, default='node_dropconnect', 
                        choices=['node_dropconnect', 'none'])
    parser.add_argument('--p', type=float, default=-1, 
                        help='the dropout rate, -1 means annealing from 1 - epoch / total epoch to 0.')

    parser.add_argument('--reg', type=float, default=5e-4, 
                        help='the regularization strength.')

    # specific to the ensemble methods
    parser.add_argument('--ensemble_n', type=int, default=1,
                        help='the number of ensembles')
    parser.add_argument('--shrinkage', type=float, default=1,
                        help='shrinkage of the boosting method')

    # parameters that do not require frequent tuning
    parser.add_argument('--back_n', type=int, default=0,
                        help='the depth of the backward network')
    parser.add_argument('--net_type', type=str, default='locally_constant', metavar='S',
                        choices=['locally_constant', 'locally_linear'])
    parser.add_argument('--hidden_dim', type=int, default=1,
                        help='the hidden dimension')
    parser.add_argument('--anneal', type=str, default='approx', choices=['interpolation', 'none', 'approx'],
                        help='whether to anneal ReLU')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'AMSGrad', 'AdamW'])
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1), 0.1 seems good for molecule classification on fingerprints, may need to tune for other datasets / tasks.')
    parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr_step_size', type=int, default=100,
                        help='How often to decrease learning by gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR is multiplied by gamma on schedule.')

    args = parser.parse_args()
    return args