import torch
import torch.nn.functional as F
from torch import nn

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_proj, cloth_dir, fallback=None):
        if img_proj is None or cloth_dir is None:
            if fallback is not None:
                return fallback, {}
            return torch.tensor(0.0), {}

        img_proj_n = F.normalize(img_proj.float(), dim=1, eps=1e-6)
        cloth_dir_n = F.normalize(cloth_dir.float(), dim=1, eps=1e-6)
        cos = (img_proj_n * cloth_dir_n).sum(dim=1)
        loss = cos.pow(2).mean()
        stats = {
            "ortho_cos_mean": cos.mean().detach(),
            "ortho_cos_max": cos.max().detach(),
        }
        return loss, stats