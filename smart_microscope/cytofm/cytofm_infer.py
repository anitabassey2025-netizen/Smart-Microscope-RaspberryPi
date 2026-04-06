import torch
import cv2
import torchvision.transforms as T
from torch import nn

from .cytofm_backbone import load_cytofm_backbone
from .abmil import ABMIL


class CytoFMABMILInference:
    """
    CytoFM backbone (ViT-B/16 @ 256) + ABMIL pooling + classifier.
    """

    def __init__(
        self,
        cytofm_weights,
        abmil_weights,
        device: str = "cpu",
        patch_size: int = 256,
        stride: int = 256,
        out_dim: int = 2,
    ):
        self.device = torch.device(device)
        self.patch = patch_size
        self.stride = stride

        # Backbone
        self.backbone = load_cytofm_backbone(cytofm_weights, device=self.device)

        # ABMIL: (dim, attn_dim=256, dropout)
        self.head = ABMIL(768, 256, 0.2)

        # Classifier (will be replaced if checkpoint defines one)
        self.classifier = nn.Linear(768, out_dim)

        # -------- LOAD WEIGHTS SAFELY ---------
        sd = torch.load(abmil_weights, map_location=self.device)

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        # Load ABMIL head weights
        head_state = self.head.state_dict()
        head_load = {k: v for k, v in sd.items()
                     if k in head_state and head_state[k].shape == v.shape}
        head_state.update(head_load)
        self.head.load_state_dict(head_state, strict=False)

        # Handle classifier weights if checkpoint contains them
        cls_keys = None
        if "classifier.weight" in sd and "classifier.bias" in sd:
            cls_keys = ("classifier.weight", "classifier.bias")
        elif "fc.weight" in sd and "fc.bias" in sd:
            cls_keys = ("fc.weight", "fc.bias")

        if cls_keys:
            w = sd[cls_keys[0]]
            b = sd[cls_keys[1]]
            ckpt_out = w.shape[0]
            if ckpt_out != self.classifier.out_features:
                self.classifier = nn.Linear(768, ckpt_out)
            cls_state = self.classifier.state_dict()
            if cls_state["weight"].shape == w.shape:
                cls_state["weight"] = w
                cls_state["bias"] = b
                self.classifier.load_state_dict(cls_state, strict=False)

        self.head.to(self.device).eval()
        self.classifier.to(self.device).eval()

        # Transforms
        self.tx = T.Compose([
            T.ToTensor(),
            T.Resize((self.patch, self.patch)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        torch.set_num_threads(2)

    # -------- TILING --------
    def _tiles(self, bgr):
        H, W = bgr.shape[:2]
        tiles, coords = [], []
        for y in range(0, H - self.patch + 1, self.stride):
            for x in range(0, W - self.patch + 1, self.stride):
                crop = bgr[y:y + self.patch, x:x + self.patch]
                if crop.shape[:2] == (self.patch, self.patch):
                    tiles.append(crop)
                    coords.append((x, y))
        return tiles, coords

    # -------- EMBEDDINGS --------
    @torch.no_grad()
    def _embed(self, tiles):
        if not tiles:
            return torch.empty(0, 768, device=self.device)
        batch = []
        for p in tiles:
            rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            batch.append(self.tx(rgb))
        x = torch.stack(batch, 0).to(self.device)

        embs = []
        for i in range(0, len(x), 16):
            embs.append(self.backbone(x[i:i+16]))
        return torch.cat(embs, 0)

    # -------- PREDICT --------
    @torch.no_grad()
    def predict(self, image_bgr):
        tiles, coords = self._tiles(image_bgr)
        embs = self._embed(tiles)

        # If no tiles, fallback to whole-image embedding
        if embs.numel() == 0:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            x = self.tx(rgb).unsqueeze(0).to(self.device)
            embs = self.backbone(x)

        bag = embs.unsqueeze(0)     # [1, N, 768]
        N = embs.shape[0]

        # ---- Call ABMIL head ----
        out = self.head(bag)

        # Normalize possible returns: (pooled, attn) or (logits, attn, pooled) or pooled-only
        if isinstance(out, tuple):
            tensors = [t for t in out if torch.is_tensor(t)]
        else:
            tensors = [out]

        pooled = None
        attn = None

        # 1) Prefer tensor with last dim = 768 → pooled
        for t in tensors:
            if t.dim() >= 2 and t.shape[-1] == 768:
                pooled = t
            if attn is None and (t.shape[-1] == N or (t.dim()==3 and t.shape[-1]==1)):
                attn = t

        # If pooled still missing but attn exists → pooled = softmax(attn) @ embs
        if pooled is None and attn is not None:
            a = attn
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.dim() == 3 and a.shape[-1] == 1:
                a = a.squeeze(-1)
            a = torch.softmax(a, dim=-1)
            pooled = a @ embs

        # If pooled is still None but exactly one tensor → if it's 768 long, use it
        if pooled is None and len(tensors) == 1:
            t = tensors[0]
            if t.numel() == 768:
                pooled = t.unsqueeze(0)

        if pooled is None:
            raise RuntimeError("ABMIL did not produce pooled embedding.")

        # Ensure shape [1, 768]
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        if pooled.shape[-1] != 768:
            raise RuntimeError(f"pooled has wrong shape {pooled.shape}, expected last dim 768")

        # ---- Classifier ----
        logits = self.classifier(pooled)

        if logits.shape[1] == 1:
            p = torch.sigmoid(logits)
            probs = torch.cat([1 - p, p], dim=1).squeeze(0).cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return {
            "probs": probs,
            "attn": None,
            "coords": coords
        }