import torch 
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index


def _reduction(loss, reduction: str = 'mean'):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


def PartialLogLikelihood(logits, fail_indicator, ties):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    ties: 'noties' or 'efron' or 'breslow'
    '''
    logL = 0
    # pre-calculate cumsum
    cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0*logL / observations

def nll_logistic_hazard(phi, idx_durations, events,
                        reduction: str = 'mean'):
    """Negative log-likelihood of the discrete time hazard parametrized model LogisticHazard [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)

def cox_ph_loss_sorted(log_h, events, eps: float = 1e-7):
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    # if events.dtype is torch.bool:
    #     events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def rank_loss(pred, rank, cens, f="RankingSVM"):
    """
    Compute the ranking loss.

    Parameters
    ----------
    pred : torch tensor
        Model prediction.
    rank : torch tensor
        Time tensor (ground truth).
    cens : torch tensor
        Event tensor.
    f : str
        Ranking function. Default: "RankingSVM".

    Returns
    -------
    loss : float
        Ranking loss.
    """
    # Find acceptable pairs

    mat_rank = rank.unsqueeze(0).repeat(rank.size(0), 1)
    # mat_rank = rank.repeat(rank.size(0), 1)
    mat_rank = mat_rank.transpose(1, 0) - mat_rank
    diagonal = torch.eye(mat_rank.size()[0])
    diagonal = diagonal.cuda()
    mat_rank = mat_rank - diagonal
    mat_rank = mat_rank >= 0
    mat_rank = mat_rank.float()
    
    cens = cens.unsqueeze(0).repeat(cens.size(0), 1)
    mat_rank = mat_rank * cens
    # print(cens.size(), mat_rank.size())

    # Differences between the pairs in the prediction
    t_diff = pred.transpose(1, 0).repeat(pred.size(0), 1)
    t_diff = t_diff.transpose(1, 0) - t_diff

    # Apply f to the time differences
    if f == "RankingSVM":
        t_diff = (t_diff - 1)
        t_diff = t_diff.clamp(0)
    elif f == "RankBoost":
        # To avoid inf with torch.exp
        t_diff = torch.clamp(t_diff, -88, 88)
        t_diff = 1. - torch.exp(-t_diff)
    elif f == "sigm":
        t_diff = torch.sigmoid(t_diff)
    elif f == "log_sigm":
        t_diff = F.logsigmoid(t_diff)
    else:
        raise NotImplementedError()

    loss = torch.sum(t_diff * mat_rank) / torch.sum(mat_rank).data.item()
    return -loss

class _Loss(torch.nn.Module):
    """Generic loss function.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

class CoxPHLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, log_h, T, E):
        idx = T.sort(descending=True)[1]
        E = E[idx]
        log_h = log_h[idx]

        return cox_ph_loss_sorted(log_h, E, self.eps)

class NLLLogistiHazardLoss(_Loss):
    """Negative log-likelihood of the hazard parametrization model.
    See `loss.nll_logistic_hazard` for details.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def forward(self, phi, idx_durations, events):
        return nll_logistic_hazard(phi, idx_durations, events, self.reduction)

class RankLoss(nn.Module):
    def __init__(self, f='RankingSVM'):
        super().__init__()
        self.f = f

    def forward(self, log_h, T, E):
        return rank_loss(log_h, T, E, self.f)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def calc_concordance_index(logits, fail_indicator, fail_time):
    """
    Compute the concordance-index value.
    Parameters:
        label_true: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
        y_pred: np.array, predictive proportional risk of network.
    Returns:
        concordance index.
    """
    hr_pred = -logits 
    ci = concordance_index(fail_time,
                            hr_pred,
                            fail_indicator)
    return ci


