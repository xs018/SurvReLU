from pycox import models
from torchtuples.tupletree import tuplefy
from models.notree import NOTree


def get_alpha(epoch, total_epoch):
    return float(epoch) / float(total_epoch)


class CoxNOSTree(models.cox._CoxPHBase):
    def __init__(self, 
                input_dim, 
                output_dim, 
                hidden_dim, 
                num_layer, 
                num_back_layer, 
                dense = False, 
                drop_type = 'none', 
                p_drop = 0.0,
                net_type = 'locally_constant', 
                approx = 'none',
                alpha = 1,
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
        
        self.p_drop = p_drop
        self.alpha = alpha

        if loss is None:
            loss = models.loss.CoxPHLoss()
        super().__init__(net, loss, optimizer, device)

    def compute_metrics(self, data, metrics=None):
        """Function for computing the loss and other metrics.

        Arguments:
            data {tensor or tuple} -- A batch of data. Typically the tuple `(input, target)`.

        Keyword Arguments:
            metrics {dict} -- A dictionary with metrics. If `None` use `self.metrics`. (default: {None})
        """
        if metrics is None:
            metrics = self.metrics
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to set `self.loss`.")

        input, target = data
        input = self._to_device(input)
        target = self._to_device(target)

        # print(input.shapes(), target.shapes())

        if self.net.net_type == 'locally_constant':
            out = self.net(*input)
        elif self.net.net_type == 'locally_linear':
            out = self.net.normal_forward(*input)
        # print(out.size())
        out = tuplefy(out)
        return {name: metric(*out, *target) for name, metric in metrics.items()}


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

    def __init__(self, input_dim, 
                output_dim, 
                hidden_dim, 
                num_layer, 
                num_back_layer, 
                dense = False, 
                drop_type = 'none', 
                p_drop = 0.0,
                net_type = 'locally_constant', 
                approx = 'none',
                alpha = 1,
                optimizer=None, 
                device=None, 
                loss=None, 
                shrink=0.):
        
        net = NOTree(input_dim, 
                output_dim, 
                hidden_dim, 
                num_layer, 
                num_back_layer, 
                dense, 
                drop_type, 
                net_type, 
                approx)
        
        self.p_drop = p_drop
        self.alpha = alpha

        super().__init__(net, optimizer, device, shrink, loss)

if __name__ == "__main__":
    pass