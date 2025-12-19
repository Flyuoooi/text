import torch
import torch.nn as nn
import torch.nn.functional as F


class CAADiscriminator(nn.Module):
    def __init__(self, dim=512):
        super(CAADiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, D] 的特征
        Returns:
            logits: [B, 1]，未经过 sigmoid
        """
        return self.net(x)
