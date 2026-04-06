
# Smart Microscope – Implementation Log & Runbook (Pi)

**Owner:** Anita Itoro Bassey  
**Host:** Raspberry Pi (CPU‑only, Python 3.11, venv)  
**Repo Root:** `~/pi_tests/smart_microscope`  

---

## TL;DR – Daily Runbook

```bash
cd ~/pi_tests
source .venv/bin/activate
export PYTHONPATH="$HOME/pi_tests"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Baselines
./run_and_log.sh smart_microscope/test_tests.py --model_name resnet
./run_and_log.sh smart_microscope/test_tests.py --model_name efficientnetv2
./run_and_log.sh smart_microscope/test_tests.py --model_name wavemix

# CytoFM
./run_and_log.sh smart_microscope/test_tests.py --model_name cytofm

# Hybrids
./run_and_log.sh smart_microscope/test_tests.py --model_name hybrid_cytofm_resnet
./run_and_log.sh smart_microscope/test_tests.py --model_name hybrid_cytofm_efficientnetv2
```

**If it feels slow/hot on Pi:**
- In `ML_models.py` hybrid constructors, use `tile_bs=4` (or `2`).
- In `hybrid_infer.py`, temporarily change tiling stride `256 → 320` to reduce tiles/image.
- Use a timeout: `timeout 45m env PYTHONPATH=$HOME/pi_tests ./run_and_log.sh ...`

---

## Environment Pins (final)

```text
numpy==1.26.4
pandas==2.1.0
pillow==12.1.0
matplotlib==3.8.0
scikit-learn==1.4.0
onnxruntime==1.16.3
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
timm==0.9.16
einops==0.7.0
wavemix==0.3.0
opencv → sudo apt install -y python3-opencv
```

> Notes: WaveMix 0.3.0 expects NumPy 1.x (imports removed internals in 2.x). ONNX Runtime 1.23.2 didn’t have aarch64/cp311 wheels; 1.16.3 does.

---

## Issues & Fixes (Chronological)

### 1) WaveMix fails under NumPy 2.x
**Symptom**: `ModuleNotFoundError: numpy.lib.function_base`  
**Fix**: Pin `numpy==1.26.4` (or patch import to `from numpy import hamming`).

### 2) Runner not forwarding CLI args / script corruption
**Fix**: Replace `run_and_log.sh` with a clean version that passes `"$@"` and logs (see code below).  
**Important**: Do **not** put an extra `--` before `--model_name`.

### 3) `test_tests.py` ignored `--model_name`
**Fix**: Proper argparse; set `MODEL_NAME = args.model_name`. Don’t reference `MODEL_NAME` before it’s defined.

### 4) CytoFM ViT positional‑embedding mismatch (197 vs 257)
**Fix**: In `cytofm_backbone.py` resize `vit.pos_embed` (keep CLS token; bilinear‑interpolate grid) before `load_state_dict(strict=False)`.

### 5) ABMIL constructor mismatch & dropout
**Fix**: `ABMIL(768, 256, 0.2)`. Add `classifier = nn.Linear(768, out_dim)` in `cytofm_infer.py`; route pooled vector → classifier.

### 6) `__pycache__` permissions
**Fix**: `sudo chown -R anita:anita ~/pi_tests`; optionally disable pyc writes via `PYTHONDONTWRITEBYTECODE=1`.

### 7) Hybrids (CytoFM + CNN late fusion)
- Extract **v_cyto (768‑D)** via CytoFM + ABMIL;
- Extract **v_cnn (D‑D)** via CNN feature extractor;  
- Concatenate → Fusion head: either simple **MLP** (Linear in=768+D → 2) or **Transformer** head (proj 2048→256, encoder layers, fc 256→2).  
- CNN embedding dims: ResNet18=**512**; EffNetV2‑S=**1280**; WaveMix (yours) **192**.

### 8) `test_data.py` – treat adapters like CytoFM
**Fix**: Adapters (CytoFM + `hybrid_cytofm_*`) should **not** load external `.pth`; use **ToTensor() only** (engine handles tiling/normalize). Plain CNNs still use ImageNet transforms + load checkpoint.

### 9) Fusion head paths
**Fix**: Use `paths.ROOT / "models" / "fusion_head_resnet.pt"` etc. (under `smart_microscope/models`).

### 10) Hybrid concat shape errors (e.g., `1x513` vs expected `1280`)
**Fix**: Force **v_cyto → [1,768]** and **v_cnn → [1,D]** before `torch.cat`.  
(Handle cases where ABMIL returns `[1,N,768]` or attention vector.)

---

## Final Code – Copy‑Ready

### A) `run_and_log.sh` (full)
```bash
#!/usr/bin/env bash
set -e
mkdir -p logs
TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/${TS}_test_tests.log"

echo "=== RUN: $1 ===" | tee "$LOGFILE"
echo "Timestamp: $(date)" | tee -a "$LOGFILE"
echo "PWD: $(pwd)" | tee -a "$LOGFILE"
echo "Python: $(python3 --version)" | tee -a "$LOGFILE"
echo "Venv: $VIRTUAL_ENV" | tee -a "$LOGFILE"

echo -e "
--- SYSTEM BEFORE ---" | tee -a "$LOGFILE"
uptime | tee -a "$LOGFILE"
free -h | tee -a "$LOGFILE"

echo -e "
--- RUNNING (/usr/bin/time -v gives peak RAM + CPU) ---" | tee -a "$LOGFILE"
if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v python "$@" 2>&1 | tee -a "$LOGFILE"
else
  python "$@" 2>&1 | tee -a "$LOGFILE"
fi
```

### B) `test_tests.py` – argparse & optional MODEL_PATH
```python
import argparse
from paths_config import MODEL_CHECKPOINTS

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = args.model_name
MODEL_PATH = str(MODEL_CHECKPOINTS[MODEL_NAME]) if MODEL_NAME in MODEL_CHECKPOINTS else None
```

### C) `cytofm_backbone.py` – positional‑embedding resize helpers (insert near top)
```python
import torch
import torch.nn.functional as F

def _infer_grid(num_tokens: int):
    side = int(round(num_tokens ** 0.5))
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer square grid for {num_tokens} tokens")
    return side, side

def _resize_pos_embed(pe: torch.Tensor, new_num_tokens: int) -> torch.Tensor:
    assert pe.ndim == 3 and pe.shape[0] == 1
    cls = pe[:, :1, :]
    grid = pe[:, 1:, :]
    N_old = grid.shape[1]
    if N_old == new_num_tokens:
        return pe
    h_old, w_old = _infer_grid(N_old)
    h_new, w_new = _infer_grid(new_num_tokens)
    C = grid.shape[-1]
    grid = grid.reshape(1, h_old, w_old, C).permute(0,3,1,2)
    grid = F.interpolate(grid, size=(h_new,w_new), mode="bilinear", align_corners=False)
    grid = grid.permute(0,2,3,1).reshape(1, h_new*w_new, C)
    return torch.cat([cls, grid], dim=1)
```

**Apply during load (before `load_state_dict`)**:
```python
if "vit.pos_embed" in sd and hasattr(m.vit, "pos_embed"):
    pe_ckpt = sd["vit.pos_embed"]
    pe_model = m.vit.pos_embed
    if pe_ckpt.shape != pe_model.shape:
        N_new = pe_model.shape[1] - 1
        sd["vit.pos_embed"] = _resize_pos_embed(pe_ckpt, N_new)
```

### D) `cytofm_infer.py` – ABMIL + classifier (final core)
```python
from torch import nn
from .abmil import ABMIL

# ABMIL head and classifier
self.head = ABMIL(768, 256, 0.2)
self.classifier = nn.Linear(768, out_dim)

# In predict(): ensure pooled vector; classifier -> logits -> probs
pooled = ...  # (use your existing robust pooling)
logits = self.classifier(pooled)  # [1, K]
```

### E) `hybrid_infer.py` – robust fusion (core pieces)
- **Auto‑detect fusion head**: simple MLP vs transformer.
- **Guarantee shapes** before concat:
```python
v_cyto = self._cyto_pooled(bgr)          # → ensure [1,768]
v_cnn  = self._cnn_feat(img)             # → ensure [1,D]

if v_cyto.ndim == 3: v_cyto = v_cyto.mean(dim=1)
elif v_cyto.ndim == 1: v_cyto = v_cyto.unsqueeze(0)
if v_cnn.ndim == 3: v_cnn = v_cnn.mean(dim=1)
elif v_cnn.ndim == 4: v_cnn = v_cnn.mean(dim=[2,3])

v_cyto = v_cyto.view(1, -1)[:, :768]
v_cnn  = v_cnn.view(1, -1)

fused = torch.cat([v_cyto, v_cnn], dim=1)   # [1, 768 + D]
logit = self.fusion(fused)                  # [1, 2]
```

### F) `ML_models.py` – hybrid getters & cases (paths fixed)
```python
def get_Hybrid_CytoFM_ResNet():
    from smart_microscope import paths_config as paths
    from smart_microscope.hybrid.hybrid_infer import HybridCytoFMAdapter
    fusion = paths.ROOT / "models" / "fusion_head_resnet.pt"
    return HybridCytoFMAdapter(
        cytofm_weights=str(paths.CYTOFM_BACKBONE),
        abmil_weights=str(paths.CYTOFM_HEAD),
        fusion_head=str(fusion),
        cnn_backbone="resnet18",
        cnn_embed_dim=512,
        device="cpu",
        tile_bs=4,
    )

def get_Hybrid_CytoFM_EfficientNetV2():
    from smart_microscope import paths_config as paths
    from smart_microscope.hybrid.hybrid_infer import HybridCytoFMAdapter
    fusion = paths.ROOT / "models" / "fusion_head_effnet.pt"
    return HybridCytoFMAdapter(
        cytofm_weights=str(paths.CYTOFM_BACKBONE),
        abmil_weights=str(paths.CYTOFM_HEAD),
        fusion_head=str(fusion),
        cnn_backbone="efficientnetv2",
        cnn_embed_dim=1280,
        device="cpu",
        tile_bs=4,
    )

# In call_model():
case "hybrid_cytofm_resnet":
    return get_Hybrid_CytoFM_ResNet()
case "hybrid_cytofm_efficientnetv2":
    return get_Hybrid_CytoFM_EfficientNetV2()
```

### G) `test_data.py` – adapter‑aware `test_model(...)` (full replacement)
```python
def test_model(test_df, model_name="resnet", model_path=None, batch_size=1, device=None):
    import os
    import numpy as np
    import torch
    from torchvision import transforms
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    name = (model_name or "").lower()
    is_adapter = (name == "cytofm") or name.startswith("hybrid_cytofm_")
    is_plain_cnn = not is_adapter

    if model_path is None:
        model_path = str(MODEL_CHECKPOINTS[model_name]) if (is_plain_cnn and model_name in MODEL_CHECKPOINTS) else None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Model Path: {model_path}")

    if is_plain_cnn:
        if model_path is None or (isinstance(model_path, str) and not os.path.exists(model_path)):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = ML_models.call_model(model_name)

    if is_plain_cnn:
        state = torch.load(model_path, map_location=device)
        # state = state.get("model_state_dict", state)
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    if is_adapter:
        transform = transforms.Compose([transforms.ToTensor()])
        if batch_size != 1:
            print("[WARN] For adapter models, forcing batch_size=1 for CPU stability.")
        batch_size = 1
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    test_loader = load_test_loader(test_df, transform, batch_size=batch_size)

    all_preds, all_labels, all_probs = [], [], []
    print("[INFO] Running inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze(dim=1).detach().cpu()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            preds = (probs >= 0.5).long()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())
            all_probs.append(probs)

    all_preds = torch.cat(all_preds) if len(all_preds) else torch.empty(0, dtype=torch.long)
    all_labels = torch.cat(all_labels) if len(all_labels) else torch.empty(0, dtype=torch.long)
    all_probs = torch.cat(all_probs) if len(all_probs) else torch.empty(0, dtype=torch.float32)

    y_true = all_labels.numpy() if all_labels.numel() else []
    y_pred = all_preds.numpy() if all_preds.numel() else []

    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) if len(y_true) else np.zeros((2, 2), dtype=int)
    specificity = _compute_specificity(cm) if cm.sum() > 0 else 0.0

    print("
[RESULTS]")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Confusion Matrix:
{cm}")

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "confusion_matrix": cm
    }
```

---

## Performance & Thermal Tips (Pi)
- **Threads**: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
- **Tile batch**: `tile_bs=4` (or `2`) in hybrid constructors.
- **Stride**: temporarily raise to `320` for faster smoke tests.
- **Timeout**: `timeout 45m env PYTHONPATH=$HOME/pi_tests ./run_and_log.sh ...`
- **Monitor**: `vcgencmd measure_temp; vcgencmd get_throttled`; reboot clears sticky throttle flags.

---

## Appendix

### Check CNN embedding dims
```bash
# ResNet18 -> 512
python - << 'EOF'
import torch
from smart_microscope.ML_models import get_Res
m = get_Res().eval()
x = torch.randn(1,3,224,224)
feat = torch.nn.Sequential(*list(m.children())[:-1])(x)
print('ResNet18 feature shape:', feat.view(1,-1).shape)
EOF

# EffNetV2-S -> 1280
python - << 'EOF'
import torch
from smart_microscope.ML_models import get_efficient
m = get_efficient().eval()
x = torch.randn(1,3,224,224)
feat = m.avgpool(m.features(x))
print('EffNetV2-S feature shape:', torch.flatten(feat,1).shape)
EOF
```

### Small quick-test dataset
```bash
mkdir -p ~/pi_tests/tmp_small/B ~/pi_tests/tmp_small/M
find ~/pi_tests/smart_microscope/CytoLabel -type f -path '*/B/*' | head -n 3 | xargs -I{} cp '{}' ~/pi_tests/tmp_small/B/
find ~/pi_tests/smart_microscope/CytoLabel -type f -path '*/M/*' | head -n 3 | xargs -I{} cp '{}' ~/pi_tests/tmp_small/M/
```

---

*End of runbook.*
