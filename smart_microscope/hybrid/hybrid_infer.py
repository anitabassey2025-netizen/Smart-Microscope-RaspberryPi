import torch
import cv2
from torch import nn
import torchvision.transforms as T
import torchvision.models as models

from smart_microscope.cytofm.cytofm_backbone import load_cytofm_backbone
from smart_microscope.cytofm.abmil import ABMIL
from smart_microscope.ML_models import get_Wavemix   # in case you later add wavemix hybrid


# ---------- transforms ----------
def _tx_224():
    return T.Compose([
        T.ToTensor(),
        T.Resize((224,224)),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225]),
    ])

def _tx_256():
    return T.Compose([
        T.ToTensor(),
        T.Resize((256,256)),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225]),
    ])


# ---------- CNN feature extractors ----------
class ResNet18Feat(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=None)
        self.fe = nn.Sequential(*list(m.children())[:-1])  # -> [B,512,1,1]

    @torch.no_grad()
    def forward(self, x):
        f = self.fe(x)
        return f.view(f.size(0), -1)  # [B,512]


class EffNetV2SFeat(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.efficientnet_v2_s(weights=None)
        self.features, self.avgpool = m.features, m.avgpool

    @torch.no_grad()
    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        return torch.flatten(f, 1)  # [B,1280]


# (WaveMix support can be added later if needed)
# class WaveMixFeat(nn.Module): ...


# ---------- Fusion builders ----------
def _looks_like_transformer(sd: dict) -> bool:
    # Any key that starts with "transformer.layers.0." is our signal
    return any(isinstance(k, str) and k.startswith("transformer.layers.0.") for k in sd.keys())

def _linear_in_dims(sd: dict):
    return { v.shape[1] for k,v in sd.items()
             if isinstance(k,str) and k.endswith("weight") and getattr(v,"ndim",0)==2 }

def _build_fusion_mlp(in_dim: int, sd: dict|None):
    # default simple MLP
    mlp = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
    )
    if isinstance(sd, dict):
        try: mlp.load_state_dict(sd, strict=False)
        except Exception: pass
    return mlp

class TransformerFusion(nn.Module):
    """proj (in_dim->256) + TransformerEncoder (L, d=256, nhead=8, ff=2048) + fc(256->2)"""
    def __init__(self, in_dim: int, sd: dict):
        super().__init__()
        self.proj = nn.Linear(in_dim, 256)
        # infer number of layers by counting
        layer_ids = set()
        for k in sd.keys():
            if isinstance(k,str) and k.startswith("transformer.layers."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_ids.add(int(parts[2]))
        num_layers = (max(layer_ids)+1) if layer_ids else 6

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(256, 2)

        # try loading by matching names
        # rename keys to match our module names if needed
        mapped = {}
        for k,v in sd.items():
            if not isinstance(k,str): continue
            if k.startswith("transformer."):
                mapped[k] = v
            elif k in ("fc.weight","fc.bias"):
                mapped[k] = v
            elif k in ("proj.weight","proj.bias"):
                mapped[k] = v
        # Load backbone modules separately for clarity
        if "proj.weight" in mapped and "proj.bias" in mapped:
            self.proj.load_state_dict({"weight": mapped["proj.weight"], "bias": mapped["proj.bias"]}, strict=False)
        # Load encoder (strict=False to ignore minor name divergences)
        enc_sd = {k: v for k,v in mapped.items() if k.startswith("transformer.")}
        try: self.transformer.load_state_dict(enc_sd, strict=False)
        except Exception: pass
        # Load classifier
        if "fc.weight" in mapped and "fc.bias" in mapped:
            try: self.fc.load_state_dict({"weight": mapped["fc.weight"], "bias": mapped["fc.bias"]}, strict=False)
            except Exception: pass

    def forward(self, fused: torch.Tensor):  # fused: [B, in_dim]
        x = self.proj(fused)           # [B,256]
        x = x.unsqueeze(1)             # [B,1,256] (seq len = 1)
        x = self.transformer(x)        # [B,1,256]
        x = x.squeeze(1)               # [B,256]
        return self.fc(x)              # [B,2]


def build_fusion_head(in_dim: int, fusion_sd: dict):
    # If the fusion state_dict contains transformer.* keys, build Transformer head
    if _looks_like_transformer(fusion_sd):
        return TransformerFusion(in_dim, fusion_sd)

    # If we see any Linear weight with in_features == in_dim, build MLP and load
    in_dims = _linear_in_dims(fusion_sd)
    if in_dim in in_dims:
        return _build_fusion_mlp(in_dim, fusion_sd)

    # Fallback: build MLP and load best-effort
    return _build_fusion_mlp(in_dim, fusion_sd)


# ---------- Hybrid Adapter ----------
class HybridCytoFMAdapter(nn.Module):
    """
    CytoFM 768-D + CNN embedding D → concat → fusion head → logits(2)
    """
    def __init__(self,
                 cytofm_weights: str,
                 abmil_weights: str,
                 fusion_head: str,
                 cnn_backbone: str = "resnet18",
                 cnn_embed_dim: int = 512,
                 device: str = "cpu",
                 tile_bs: int = 8):
        super().__init__()
        self.device = torch.device(device)
        self.tile_bs = tile_bs

        # CytoFM backbone + ABMIL
        self.backbone = load_cytofm_backbone(cytofm_weights, device=self.device)
        self.abmil    = ABMIL(768, 256, 0.2).to(self.device).eval()

        # Load ABMIL weights (best-effort)
        try:
            sd = torch.load(abmil_weights, map_location=self.device)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
            head_state = self.abmil.state_dict()
            head_load  = {k:v for k,v in sd.items() if k in head_state and v.shape == head_state[k].shape}
            head_state.update(head_load)
            self.abmil.load_state_dict(head_state, strict=False)
        except Exception:
            pass

        # CNN feature extractor
        self.cnn_name = cnn_backbone.lower()
        if self.cnn_name == "resnet18":
            self.cnn = ResNet18Feat().to(self.device).eval()
        elif self.cnn_name == "efficientnetv2":
            self.cnn = EffNetV2SFeat().to(self.device).eval()
        else:
            raise ValueError(f"Unknown cnn_backbone {cnn_backbone}")

        self.tx256, self.tx224 = _tx_256(), _tx_224()

        # Fusion head (auto-detect MLP vs Transformer)
        self.in_dim = 768 + cnn_embed_dim
        fusion_sd = torch.load(fusion_head, map_location=self.device)
        if isinstance(fusion_sd, dict) and "state_dict" in fusion_sd:
            fusion_sd = fusion_sd["state_dict"]
        self.fusion = build_fusion_head(self.in_dim, fusion_sd).to(self.device).eval()

        torch.set_num_threads(2)

    # ------ helpers ------
    @torch.no_grad()
    def _tiles(self, bgr):
        H,W = bgr.shape[:2]
        tiles=[]
        for y in range(0, H-256+1, 256):
            for x in range(0, W-256+1, 256):
                crop = bgr[y:y+256, x:x+256]
                if crop.shape[:2] == (256,256): tiles.append(crop)
        return tiles

    @torch.no_grad()
    def _cyto_pooled(self, bgr):
        # --- tile & embed ---
        tiles = self._tiles(bgr)
        if not tiles:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = self.tx256(rgb).unsqueeze(0).to(self.device)
            embs = self.backbone(x)                      # [1,768]
        else:
            batch = []
            for p in tiles:
                rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
                batch.append(self.tx256(rgb))
            x = torch.stack(batch, 0).to(self.device)    # [N,3,256,256]
            embs = []
            for i in range(0, x.size(0), self.tile_bs):
                embs.append(self.backbone(x[i:i+self.tile_bs]))
            embs = torch.cat(embs, 0)                    # [N,768]

        bag = embs.unsqueeze(0)                           # [1,N,768] or [1,1,768]
        out = self.abmil(bag)

        # --- Parse ABMIL outputs robustly ---
        pooled, attn = None, None
        tensors = out if isinstance(out, (tuple, list)) else [out]

        # Find pooled (last dim 768) and attention ([1,N], [N], or [1,N,1])
        for t in tensors:
            if torch.is_tensor(t):
                if t.ndim >= 2 and t.shape[-1] == 768 and pooled is None:
                    pooled = t
                if attn is None and (
                    (t.ndim == 2 and t.shape[0] == 1) or
                    (t.ndim == 1) or
                    (t.ndim == 3 and t.shape[-1] == 1)
                ):
                    attn = t

        # If ABMIL returned a sequence [1,N,768], reduce using attn if available; else mean over N.
        if pooled is not None:
            if pooled.ndim == 3 and pooled.shape[1] > 1:     # [1,N,768]
                if attn is not None:
                    a = attn
                    if a.ndim == 3 and a.shape[-1] == 1:
                        a = a.squeeze(-1)                    # [1,N]
                    if a.ndim == 1:
                        a = a.unsqueeze(0)                   # [1,N]
                    a = torch.softmax(a, dim=-1)             # [1,N]
                    if embs.ndim == 2 and embs.shape[0] == a.shape[-1]:
                        pooled_vec = a @ embs                # [1,768]
                    else:
                        pooled_vec = pooled.mean(dim=1)      # [1,768]
                else:
                    pooled_vec = pooled.mean(dim=1)          # [1,768]
            elif pooled.ndim == 2 and pooled.shape[-1] == 768:  # [1,768]
                pooled_vec = pooled
            elif pooled.ndim == 1 and pooled.numel() == 768:    # [768]
                pooled_vec = pooled.unsqueeze(0)                 # [1,768]
            else:
                # last fallback
                pooled_vec = pooled.reshape(1, -1)[:, :768]
        else:
            # No pooled found; compute via attention if possible; else mean over bag
            if attn is not None and embs.ndim == 2:
                a = attn
                if a.ndim == 3 and a.shape[-1] == 1:
                    a = a.squeeze(-1)
                if a.ndim == 1:
                    a = a.unsqueeze(0)
                a = torch.softmax(a, dim=-1)
                pooled_vec = a @ embs                          # [1,768]
            else:
                pooled_vec = (bag if bag.ndim == 3 else embs.unsqueeze(0)).mean(dim=1)

        # Ensure final shape [1,768]
        if pooled_vec.ndim == 1:
            pooled_vec = pooled_vec.unsqueeze(0)
        if pooled_vec.shape[-1] != 768:
            pooled_vec = pooled_vec.view(1, -1)[:, :768]

        return pooled_vec  # [1,768]

    @torch.no_grad()
    def _cnn_feat(self, rgb_uint8):
        x = self.tx224(rgb_uint8).unsqueeze(0).to(self.device)
        return self.cnn(x)  # [1,D]

    # ------ forward ------
    @torch.no_grad()
    def forward(self, x):  # x: [B,3,H,W]
        import numpy as np
        B = x.shape[0]
        outs = []
        for i in range(B):
            # -- build RGB uint8 image + BGR for CytoFM --
            img = x[i].detach().cpu().numpy()           # [3,H,W] float
            img = np.transpose(img, (1,2,0))            # [H,W,3] RGB
            img = (img * 255.0).clip(0,255).astype('uint8')
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # -- features --
            v_cyto = self._cyto_pooled(bgr)   # pooled CytoFM feature (should be [1,768])
            v_cnn  = self._cnn_feat(img)      # CNN feature             (should be [1,D])

            # -------- SAFETY NORMALIZATION (put this EXACTLY here) --------
            if v_cyto.ndim == 3:              # [1,N,768] -> [1,768]
                v_cyto = v_cyto.mean(dim=1)
            elif v_cyto.ndim == 1:            # [768] or [N] -> [1,*]
                v_cyto = v_cyto.unsqueeze(0)
            if v_cnn.ndim == 3:
                v_cnn = v_cnn.mean(dim=1)
            elif v_cnn.ndim == 4:
                v_cnn = v_cnn.mean(dim=[2,3])
            v_cyto = v_cyto.view(1, -1)[:, :768]
            v_cnn  = v_cnn.view(1, -1)
            # ---------------------------------------------------------------

            fused = torch.cat([v_cyto, v_cnn], dim=1)  # [1, 768 + D]
            logit = self.fusion(fused)                 # [1,2]
            outs.append(logit.squeeze(0))

        return torch.stack(outs, dim=0)  # [B,2]