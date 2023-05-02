import torch
import torch.nn as nn


class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()
        att = result['frame']
        t = att.size(1)
        anomaly = torch.topk(att, t // 16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        cost = anomaly_loss
        loss['total_loss'] = round(cost.item(), 4)
        return cost, loss

def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss

class AD_rtfm_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()
        att = result['frame']
        t = att.size(1)
        anomaly = torch.topk(att, t // 16 + 1, dim=-1)[0].mean(-1)
        rtfm_loss = sparsity(anomaly[anomaly.size(0) // 2:], 8e-3) + smooth(anomaly, 8e-4)
        anomaly_loss = self.bce(anomaly, _label)
        cost = anomaly_loss + rtfm_loss
        loss['total_loss'] = round(cost.item(), 4)
        loss['cls_loss'] = round(anomaly_loss.item(), 4)
        loss['rtfm_loss'] = round(rtfm_loss.item(), 4)
        return cost, loss
