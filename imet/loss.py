import torch
import torch.nn as nn
import torch.nn.functional as F

# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109


class FocalLoss(nn.Module):
    """Adapted from: https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
    F.logsimoid used as in https://gist.github.com/AdrienLE/bf31dfe94569319f6e47b2de8df13416#file-focal_dice_1-py
    """

    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        assert alpha > 0 and alpha < 1
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        y = y.float()
        pt_log = F.logsigmoid(-x * (y * 2 - 1))
        # w = alpha if t > 0 else 1-alpha
        at = (self.alpha * y + (1-self.alpha) * (1-y)) * 2
        w = at * (pt_log * self.gamma).exp()
        # Don't calculate gradients of the weights
        w = w.detach()
        return F.binary_cross_entropy_with_logits(x, y, w, reduction="mean")

    def __str__(self):
        return f"<Focal Loss alpha={self.alpha} gamma={self.gamma}>"


class FbetaLoss(nn.Module):
    def __init__(self, beta=1):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / \
            (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss
