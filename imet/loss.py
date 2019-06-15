import torch
import torch.nn as nn
import torch.nn.functional as F

# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, preds, target):
        if not (target.size() == preds.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), preds.size()))
        max_val = (-preds).clamp(min=0)
        loss = (
            preds - preds * target + max_val +
            ((-max_val).exp() + (-preds - max_val).exp()).log()
        )

        invprobs = F.logsigmoid(-preds * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


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
