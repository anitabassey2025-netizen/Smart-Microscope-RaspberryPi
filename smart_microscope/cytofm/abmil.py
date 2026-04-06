import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    """
    Attention-based MIL (Ilse et al.):

    Inputs:
      H: (N, D) patch embeddings
    Outputs:
      logit: (1,) binary logit
      attn: (N,) attention weights over patches
    """
    def __init__(self, dim: int, attn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1)
        )
        self.classifier = nn.Linear(dim, 1)

    def forward(self, H: torch.Tensor):
        # H: (N, D)
        a = self.attn(H)              # (N, 1)
        a = torch.softmax(a.squeeze(1), dim=0)  # (N,)
        z = torch.sum(H * a.unsqueeze(1), dim=0)  # (D,)
        logit = self.classifier(z).squeeze(0)     # (1,) -> scalar tensor
        return logit, a
