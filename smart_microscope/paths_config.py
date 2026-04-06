# paths_config.py  (Pi version)
from pathlib import Path

# Base directory on the Pi
ROOT = Path.home() / "pi_tests" / "smart_microscope"

#CYTOFM
CYTOFM_BACKBONE = ROOT / "models" / "cytofm_weights.pth"       # adjust filename if needed
CYTOFM_HEAD     = ROOT / "models" / "best_model_Split_1.pt"    # adjust if needed

# Subfolders
MODELS_DIR = ROOT / "models"
CYTO_DIR   = ROOT / "CytoLabeled"  # contains B/ and M/

# Default checkpoints — rename to match your actual files in MODELS_DIR

# If you keep a dict of checkpoints:
MODEL_CHECKPOINTS = {
    "resnet":        ROOT / "models" / "resnet_best_model_Split_1.pth",
    "efficientnetv2":ROOT / "models" / "efficientnetv2_best_model_Split_1.pth",
    "wavemix":       ROOT / "models" / "wavemix_best_model_Split_1.pth",
    "hybrid":        ROOT / "models" / "hybrid_best_model_Split_1.pth",
    # For cytofm, the adapter loads its own two files, so this value is not actually used.
    "cytofm":        CYTOFM_HEAD,
}

# Optional: where to save Grad-CAM or outputs (kept local on Pi)
OUTPUTS_DIR = ROOT / "outputs"
