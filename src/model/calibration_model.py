import torch

class CalibrationModel(torch.nn.Module):
    """ Adds temperature scaling parameters to trained model"""

    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature
        