import torch.nn.functional as F
import torch
import torch.nn as nn
def nll_loss(output, target):
    return F.nll_loss(output, target)


def bceloss(output, target):
    a=nn.BCELoss()
    return a(output, target)


class logMAEloss(nn.Module):
    def _init_(self):
        #     def _init_(self,inputs,targets):
        super(logMAEloss, self)._init_()
        #         self.inputs = inputs
        #         self.targets = targets
        return

    def forward(self, inputs, targets):
        # ‑log((‑x)+1)
        mae = torch.abs(inputs - targets)
        loss = -torch.log((-mae) + 1.0)
        return torch.mean(loss)


# In[2]:
def cross_entropy_loss_HED(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.0 * num_positive / (num_positive + num_negative)
    #     mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    #     mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

