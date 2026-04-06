from pathlib import Path
from PIL import Image
import sys
import random

import torch
import torch.nn as nn
import torchvision.transforms as T

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from abmil import ABMIL

# -----------------------
# Paths / settings
# -----------------------
PROJECT_ROOT = Path("/gpfs/projects/e33107/BC_ML_backup")
IBOT_DIR = PROJECT_ROOT / "ibot"

RUN_TAG = "cytofm_run02_full10x"
RUNS_DIR = PROJECT_ROOT / "bm_experiments" / "runs" / "cytofm" / RUN_TAG

TEST_DIR = PROJECT_ROOT / "cytotest"
B_DIR = TEST_DIR / "B"
M_DIR = TEST_DIR / "M"

WEIGHTS_PATH = PROJECT_ROOT / "weights" / "cytofm_weights.pth"

PATCH = 224
STRIDE = 224
ATTN_DIM = 256
SEED = 0

# -----------------------
# iBOT import
# -----------------------
sys.path.append(str(IBOT_DIR))
from models.vision_transformer import vit_base  # type: ignore


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


to_tensor = T.Compose([
    T.ToTensor(),
])


def list_images(folder: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return sorted(files)


def extract_patches(img_t: torch.Tensor, patch=PATCH, stride=STRIDE):
    C, H, W = img_t.shape
    pad_h = (patch - (H % patch)) % patch
    pad_w = (patch - (W % patch)) % patch
    img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h))

    patches = []
    _, H2, W2 = img_t.shape
    for y in range(0, H2 - patch + 1, stride):
        for x in range(0, W2 - patch + 1, stride):
            patches.append(img_t[:, y:y+patch, x:x+patch])
    return torch.stack(patches, dim=0)  # (N,3,patch,patch)


def load_cytofm_backbone(device: torch.device):
    model = vit_base(patch_size=16)
    ckpt = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "student" in ckpt:
        student = ckpt["student"]
        state = student["model"] if isinstance(student, dict) and "model" in student else student
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        k2 = k
        for prefix in ("module.", "backbone.", "module.backbone.", "student.", "teacher.", "student.backbone.", "teacher.backbone."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        cleaned[k2] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def predict_one(backbone, mil, device, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    img_t = to_tensor(img)
    patches = extract_patches(img_t)  # (N,3,PATCH,PATCH)

    feats_list = []
    for i in range(0, patches.shape[0], 8):
        batch = patches[i:i+8].to(device)
        feats = backbone(batch)
        feats_list.append(feats.detach())
    H = torch.cat(feats_list, dim=0)  # (N,768) on GPU

    logit, _ = mil(H)
    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= 0.5 else 0
    return prob, pred


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    b_files = list_images(B_DIR)
    m_files = list_images(M_DIR)
    paths = b_files + m_files
    y_true = [0]*len(b_files) + [1]*len(m_files)

    assert len(paths) > 0, "No test images found."
    print(f"Found test images: {len(b_files)} benign, {len(m_files)} malignant")

    backbone = load_cytofm_backbone(device)

    # evaluate each fold model separately
    rows = []
    for k in range(1, 6):
        pt_path = RUNS_DIR / f"best_model_Split_{k}.pt"
        if not pt_path.exists():
            print(f"[SKIP] missing: {pt_path}")
            continue

        mil = ABMIL(dim=768, attn_dim=ATTN_DIM, dropout=0.1).to(device)
        mil.load_state_dict(torch.load(pt_path, map_location=device))
        mil.eval()

        probs = []
        preds = []
        for p in paths:
            prob, pred = predict_one(backbone, mil, device, p)
            probs.append(prob)
            preds.append(pred)

        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
        cm = [[tn, fp], [fn, tp]]
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append({
            "Model": "cytofm",
            "Split": k,
            "Predictions": preds,
            "Labels": y_true,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_score": f1,
            "Specificity": spec,
            "Confusion_Matrix": cm,
        })

        print(f"[Split {k}] acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} spec={spec:.3f}")

    # save CSV
    out_dir = PROJECT_ROOT / "bm_experiments" / "model_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{RUN_TAG}_cytotest_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[WROTE] {out_csv}")


if __name__ == "__main__":
    main()