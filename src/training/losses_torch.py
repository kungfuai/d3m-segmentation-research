import math

import torch
from scipy.special import lambertw
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from src.training.lambertw_torch import lambertw

class BinaryFocalLoss(torch.nn.Module):
    """ from https://github.com/qubvel/segmentation_models"""

    def __init__(self, gamma=2.0, alpha=0.25, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pr, gt):
        pr = torch.clamp(pr, self.eps, 1-self.eps)

        loss_1 = - gt * (self.alpha * torch.pow(1-pr, self.gamma)) * torch.log(pr)
        loss_0 = - (1 - gt) * ((1-self.alpha) * torch.pow(pr, self.gamma)) * torch.log(1-pr)
        loss = loss_0 + loss_1
        return loss

class SuperLoss(torch.nn.Module):
    """ https://papers.nips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf"""

    def __init__(self, base_loss, lam=1.0, tau=0.0, device=torch.device('cpu')):
        super().__init__()
        self.base_loss = base_loss
        self.lam = lam
        self.tau = tau
        self.device = device
    
    def forward(self, pr, gt):
        input_loss = self.base_loss(pr, gt)

        beta_0 = torch.ones(input_loss.shape) * (-2/(math.e+0.08))
        beta_0 = beta_0.to(self.device)
        beta = (input_loss - self.tau) / self.lam
        y = 0.5 * torch.max(beta_0, beta)
        
        sigma = torch.exp(-lambertw(y))
        loss = sigma * (input_loss - self.tau) + self.lam * torch.log(sigma) ** 2
        return loss

if __name__ == "__main__":
    """ plot SuperLoss as f(input_loss - tau) for different lamda """

    lams = [0.01, 0.1, 1.0, 10.0]
    l_minus_taus = [-2, -1, 0, 1, 2]

    class NaiveLoss(torch.nn.Module):
        def __init__(self, loss):
            super().__init__()
            self.loss = torch.tensor([loss])
        def forward(self, pr, gt):
            return self.loss
        
    data = pd.DataFrame({'normalized input loss': [], 'super loss': [], 'lam': []})

    for lam in lams:
        for lmt in l_minus_taus:
            super_loss = SuperLoss(NaiveLoss(lmt), lam=lam, tau=0)
            loss = super_loss(torch.tensor([0]), torch.tensor([0])).mean()
            data = data.append(
                {'normalized input loss': lmt, 'super loss': loss.item(), 'lam': str(lam)},
                ignore_index=True
            )

    sns.lineplot(
        x="normalized input loss", 
        y="super loss",
        hue="lam",
        data=data,
    )
    plt.savefig('test_superloss.png')

