import torch
import torch.nn as nn


CLS_WEIGHT = 100
REG_WEIGHT = 10


class ConditionalMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.BCEWithLogitsLoss()
        self.reg = nn.L1Loss(reduction="none")


    def forward(self, preds, targets):
        logits = preds[:, 0]
        labels = targets[:, 0].float()


        loss_cls = CLS_WEIGHT * self.cls(logits, labels)


        mask = labels == 1
        if mask.sum() > 0:
            loss_reg = REG_WEIGHT * self.reg(preds[mask, 1:], targets[mask, 1:]).mean()
        else:
            loss_reg = torch.tensor(0.0, device=preds.device)


        return loss_cls + loss_reg