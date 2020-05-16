import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#def cross_entropy_loss_BDCN(prediction,label):
#    label=label.long()
#    mask=label.float()
def cross_entropy_loss2d(prediction, label, cuda=False):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    #label = label.long()
    #mask = label.float()
    n, c, h, w = prediction.size()
    #print("prediction.size() ", prediction.size)
    weights = np.zeros((n, c, h, w))
    for i in range(n):
       # t = label[i, :, :, :].cpu().data.numpy()
       t = label[i, :, :, :].cpu().data.numpy()
       pos = (t == 1).sum()
       neg = (t == 0).sum()
       valid = neg + pos
       weights[i, t == 1] = neg * 1.0 / valid
       #weights[i, t == 1] = neg * 0.5 / valid
       weights[i, t == 0] = pos * 1.1 / valid
    weights = torch.Tensor(weights).to(device)

    if cuda:
        weights = weights.cuda()
    inputs = F.sigmoid(prediction)
    loss = nn.BCELoss(weights, size_average=False)(inputs.float(), label.float())
    #cost = torch.nn.functional.binary_cross_entropy(
    #       inputs.float(), label.float(), weights, reduce=False)
    #print("cross_entropy_loss2d loss:", loss)
    return loss

def cross_entropy_loss_BDCN(prediction, label):
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
