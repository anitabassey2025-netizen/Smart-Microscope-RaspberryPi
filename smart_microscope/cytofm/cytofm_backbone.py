# smart_microscope/cytofm/cytofm_backbone.py

import torch
import timm
from torch import nn
from collections import OrderedDict
import torch
import torch.nn.functional as F

def _infer_grid(num_tokens: int):
    """Infer a square HxW grid from token count (no CLS)."""
    side = int(round(num_tokens ** 0.5))
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer square grid for {num_tokens} tokens")
    return side, side

def _resize_pos_embed(pe: torch.Tensor, new_num_tokens: int) -> torch.Tensor:
    """
    Resize ViT positional embedding from [1, N_old+1, C] -> [1, N_new+1, C].
    Keeps the CLS token intact; bilinear-interpolates the patch grid.
    """
    assert pe.ndim == 3 and pe.shape[0] == 1, "pos_embed must be [1, N, C]"
    cls_tok = pe[:, :1, :]          # [1,1,C]
    grid_tok = pe[:, 1:, :]         # [1,N_old,C]
    N_old = grid_tok.shape[1]
    N_new = new_num_tokens
    if N_old == N_new:
        return pe  # nothing to do

    # Old/new grid sizes
    h_old, w_old = _infer_grid(N_old)
    h_new, w_new = _infer_grid(N_new)

    # [1,N_old,C] -> [1,C,h_old,w_old]
    C = grid_tok.shape[-1]
    grid_tok = grid_tok.reshape(1, h_old, w_old, C).permute(0, 3, 1, 2)

    # Bilinear resize to new grid
    grid_tok = F.interpolate(grid_tok, size=(h_new, w_new), mode="bilinear", align_corners=False)

    # [1,C,h_new,w_new] -> [1,N_new,C]
    grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(1, h_new*w_new, C)

    return torch.cat([cls_tok, grid_tok], dim=1)


class FrozenCytoFMViT(nn.Module):
    """
    ViT-B/16 backbone returning a pooled 768-D vector per 256x256 tile.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", embed_dim: int = 768, img_size: int = 256):
        super().__init__()
        # Ask timm for a ViT-B/16 configured for 256x256 input and CLS pooling
        self.vit = timm.create_model(
            model_name,
            pretrained=False,
            img_size=img_size,
            num_classes=0,        # drop classifier head
            global_pool="token",  # CLS pooling
        )
        self.embed_dim = embed_dim
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)  # sometimes [B,C] or [B,T,C]
        if feats.ndim == 3:
            feats = feats[:, 0, :]            # take CLS token
        return feats


def _unwrap_and_remap_state(sd: dict) -> OrderedDict:
    """
    Unwrap common iBOT/DINO-style checkpoints and remap keys to match our timm model.
    - Prefer the 'teacher' branch (more stable), then 'student'
    - Strip wrappers: 'state_dict', 'backbone', 'model', 'module'
    - Prefix 'vit.' if missing so keys match self.vit.*
    """
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # choose teacher > student if present
    if isinstance(sd, dict) and "teacher" in sd and isinstance(sd["teacher"], (dict, OrderedDict)):
        sd = sd["teacher"]
    elif isinstance(sd, dict) and "student" in sd and isinstance(sd["student"], (dict, OrderedDict)):
        sd = sd["student"]

    # unwrap "backbone" or "model" nests
    for key in ("backbone", "model", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], (dict, OrderedDict)):
            sd = sd[key]

    new_sd = OrderedDict()
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        # strip common prefixes
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        # timm ViT expects attributes under "vit.*"
        if not k.startswith("vit."):
            k = "vit." + k
        new_sd[k] = v
    return new_sd


def load_cytofm_backbone(weights_path: str, device: str = "cpu") -> FrozenCytoFMViT:
    m = FrozenCytoFMViT(model_name="vit_base_patch16_224", embed_dim=768, img_size=256)
    raw = torch.load(weights_path, map_location=device)

    # If raw is a full training blob, unwrap + remap to ViT keys
    if isinstance(raw, dict):
        sd = _unwrap_and_remap_state(raw)
    else:
        sd = raw

    # Ensure keys are prefixed with "vit." to match timm model attributes
    if isinstance(sd, dict) and all((isinstance(k, str) and not k.startswith("vit.")) for k in sd.keys()):
        sd = { (f"vit.{k}" if isinstance(k, str) and not k.startswith("vit.") else k): v for k, v in sd.items() }

    # --- NEW: resize positional embedding if token count changed ---
    with torch.no_grad():
        if "vit.pos_embed" in sd and hasattr(m.vit, "pos_embed"):
            pe_ckpt = sd["vit.pos_embed"]           # [1, N_old+1, C]
            pe_model = m.vit.pos_embed              # [1, N_new+1, C]
            if tuple(pe_ckpt.shape) != tuple(pe_model.shape):
                N_new = pe_model.shape[1] - 1       # exclude CLS
                sd["vit.pos_embed"] = _resize_pos_embed(pe_ckpt, new_num_tokens=N_new)

    # Load with strict=False so benign diffs don't block
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[CytoFM] Missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[CytoFM] Unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}")

    m.to(device).eval()
    return m