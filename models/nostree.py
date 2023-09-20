from pycox import models
from torchtuples.tupletree import tuplefy
import torchtuples.callbacks as cb
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
                loss=None,
                L1_reg=1e-4):

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
        self.L1_reg = L1_reg

        if loss is None:
            loss = models.loss.CoxPHLoss()
        super().__init__(net, loss, optimizer, device)

    def fit_dataloader(
        self, dataloader, epochs=1, callbacks=None, verbose=True, metrics=None, val_dataloader=None
    ):
        """Fit a dataloader object.
        See 'fit' for tensors and np.arrays.

        Arguments:
            dataloader {dataloader} -- A dataloader that gives (input, target).

        Keyword Arguments:
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})

        Returns:
            TrainingLogger -- Training log
        """
        self._setup_train_info(dataloader)
        self.metrics = self._setup_metrics(metrics)
        self.log.verbose = verbose
        self.val_metrics.dataloader = val_dataloader
        if callbacks is None:
            callbacks = []
        self.callbacks = cb.TrainingCallbackHandler(
            self.optimizer, self.train_metrics, self.log, self.val_metrics, callbacks
        )
        self.callbacks.give_model(self)

        stop = self.callbacks.on_fit_start()
        for _ in range(epochs):
            if stop:
                break
            stop = self.callbacks.on_epoch_start()
            if stop:
                break
            for data in dataloader:
                stop = self.callbacks.on_batch_start()
                if stop:
                    break
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(data, self.metrics)
                self.batch_loss = self.batch_metrics["loss"]
                L1_loss = 0.0
                for i, (name, param) in enumerate(self.net.layer.named_parameters()):
                    if ('weight' in name) and i > 0:
                        L1_loss += param.data.abs().sum()

                self.batch_loss += self.L1_reg * L1_loss
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.batch_loss.backward()
                stop = self.callbacks.before_step()
                if stop:
                    break
                self.optimizer.step()
                stop = self.callbacks.on_batch_end()
                if stop:
                    break
            else:
                stop = self.callbacks.on_epoch_end()
        self.callbacks.on_fit_end()
        return self.log

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