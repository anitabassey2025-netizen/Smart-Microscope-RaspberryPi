# new_import.py (Pi-ready)
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _label_from_name(name: str):
    n = name.lower()
    if n in {"b", "benign"} or "benign" in n:
        return "B"
    if n in {"m", "malignant"} or "malign" in n:
        return "M"
    # allow explicit B/M folder names
    if n == "b":
        return "B"
    if n == "m":
        return "M"
    return None

def _derive_id_from_name(name: str):
    # Try to grab some digits; else use the stem
    m = re.findall(r"\d+", name)
    if m:
        return m[0]
    return re.sub(r"\W+", "", name)[:16]  # simple safe fallback

def _norm_mag(m: str | None):
    if not m:
        return "10X"
    m = m.upper().replace(" ", "")
    if m in {"10X", "X10"}:
        return "10X"
    if m in {"40X", "X40"}:
        return "40X"
    # If some odd token, keep as-is but uppercase
    return m

def import_image_data(folder_path: str | os.PathLike):
    """
    Returns a DataFrame with columns:
      image_path, label(B|M), mag(10X|40X|...), ID, section, width, height
    Works with:
      - Flat:  CytoLabeled/B/*.png, CytoLabeled/M/*.jpg
      - Nested benign/malignant with patient/section/mag folders (original Quest style)
    """
    root = Path(folder_path)
    rows = []

    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    # First-level: expect B/M or benign/malign
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        label = _label_from_name(label_dir.name)
        if label is None:
            # Skip unrelated folder names
            continue

        # MODE A: flat layout -> images live directly under B/ or M/
        flat_images = [p for p in label_dir.iterdir() if p.is_file() and _is_image(p)]
        if flat_images:
            for img in flat_images:
                try:
                    with Image.open(img) as im:
                        w, h = im.size
                except Exception:
                    w = h = None
                rows.append({
                    "image_path": str(img),
                    "label": label,
                    "mag": "10X",                # default (your Pi set is all 10X)
                    "ID": _derive_id_from_name(img.stem),  # fake ID from filename
                    "section": None,
                    "width": w, "height": h
                })
            # Done with flat mode for this label
            continue

        # MODE B: nested (Quest-like)
        for patient_dir in sorted([p for p in label_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]):
            pid = _derive_id_from_name(patient_dir.name)

            # one filler folder (e.g., coloration) or directly section/mag
            children = [p for p in patient_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
            if not children:
                continue

            # If there is exactly one "filler" level, descend; else use current level
            level1 = children
            if len(children) == 1:
                level2 = [p for p in children[0].iterdir() if p.is_dir() and not p.name.startswith(".")]
                if level2:
                    level1 = level2

            # At this level we may have MAG folders (ending with X) or section folders containing MAG
            for d in level1:
                if d.name.upper().endswith("X"):  # MAG folder
                    mag = _norm_mag(d.name)
                    section = None
                    for img in sorted([p for p in d.iterdir() if p.is_file() and _is_image(p)]):
                        try:
                            with Image.open(img) as im:
                                w, h = im.size
                        except Exception:
                            w = h = None
                        rows.append({
                            "image_path": str(img),
                            "label": label,
                            "mag": mag,
                            "ID": pid,
                            "section": section,
                            "width": w, "height": h
                        })
                else:
                    # Treat as section folder -> look for MAG children inside
                    section = d.name
                    mag_folders = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")]
                    if not mag_folders:
                        # maybe images are directly here (rare)
                        for img in sorted([p for p in d.iterdir() if p.is_file() and _is_image(p)]):
                            try:
                                with Image.open(img) as im:
                                    w, h = im.size
                            except Exception:
                                w = h = None
                            rows.append({
                                "image_path": str(img),
                                "label": label,
                                "mag": "10X",
                                "ID": pid,
                                "section": section,
                                "width": w, "height": h
                            })
                        continue

                    for mag_dir in mag_folders:
                        mag = _norm_mag(mag_dir.name)
                        for img in sorted([p for p in mag_dir.iterdir() if p.is_file() and _is_image(p)]):
                            try:
                                with Image.open(img) as im:
                                    w, h = im.size
                            except Exception:
                                w = h = None
                            rows.append({
                                "image_path": str(img),
                                "label": label,
                                "mag": mag,
                                "ID": pid,
                                "section": section,
                                "width": w, "height": h
                            })

    df = pd.DataFrame(rows, columns=["image_path","label","mag","ID","section","width","height"])
    return df

def extract_mag(df: pd.DataFrame):
    # Normalize mag if present; if not, assume 10X
    if "mag" not in df.columns:
        df = df.copy()
        df["mag"] = "10X"
    mags = df["mag"].astype(str).str.upper().str.replace(" ", "", regex=False)
    x10 = df[mags.isin({"10X", "X10"})].copy()
    x40 = df[mags.isin({"40X", "X40"})].copy()
    return x10, x40

def split_data(df: pd.DataFrame, ratio: float = 0.2, seed: int = 67):
    # Split by unique IDs to avoid leakage. If you have dummy IDs, this still groups images per file stem.
    ids = df["ID"].astype(str).unique()
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(ids) * ratio)) if len(ids) > 1 else 1
    test_ids = set(rng.choice(ids, size=n_test, replace=False))
    test_df = df[df["ID"].astype(str).isin(test_ids)].copy()
    train_df = df[~df["ID"].astype(str).isin(test_ids)].copy()
    # Safety: no overlap
    assert set(test_df["ID"]).isdisjoint(set(train_df["ID"]))
    return test_df, train_df