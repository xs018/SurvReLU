from pycox import models

from models.notree import NOTree


class CoxNOSTree(models.cox._CoxPHBase):
    def __init__(self, 
                input_dim, 
                output_dim, 
                hidden_dim, 
                num_layer, 
                num_back_layer, 
                dense = False, 
                drop_type = 'none', 
                net_type = 'locally_constant', 
                approx = 'none',
                optimizer=None, 
                device=None, 
                loss=None):

        net = NOTree(input_dim, 
                output_dim, 
                hidden_dim, 
                num_layer, 
                num_back_layer, 
                dense, 
                drop_type, 
                net_type, 
                approx)
        
        if loss is None:
            loss = models.loss.CoxPHLoss()
        super().__init__(net, loss, optimizer, device)


class CoxCCNOSTree(models.cox._CoxPHBase, models.cox_cc._CoxCCBase):
    """Cox proportional hazards model parameterized with a neural net and
    trained with case-control sampling [1].
    This is similar to DeepSurv, but use an approximation of the loss function.
    
    Arguments:
        net {torch.nn.Module} -- A PyTorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    References:
    [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
        Time-to-event prediction with neural networks and Cox regression.
        Journal of Machine Learning Research, 20(129):1–30, 2019.
        http://jmlr.org/papers/v20/18-424.html
    """
    make_dataset = models.data.CoxCCDataset


if __name__ == "__main__":
    pass