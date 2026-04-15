import torch
from baseline_loss_functions.NCELoss import NCELoss
from baseline_loss_functions.AUELoss import AUELoss

class NCEandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=6, q=1.5):
        super(NCEandAUE, self).__init__()

        if num_classes == 10:
            a = 6
            q = 1.5
        elif num_classes == 100:
            a = 6
            q = 3
            alpha = 10
            beta = 0.015

        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.aue = AUELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)
