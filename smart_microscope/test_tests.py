'''import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import new_import
import pandas as pd
import test_data as test

test_df = new_import.import_image_data(r"/home/anita/pi_tests/smart_microscope/CytoLabeled")
x10_df, x40_df = new_import.extract_mag(test_df)
test_x10_df,train_x10_df = new_import.split_data(x10_df)
print(len(test_x10_df))
print(len(train_x10_df))



# NOW TEST WITH THE FILTERED DATA
test.test_model(test_x10_df, model_path=
    r"/home/anita/pi_tests/smart_microscope/models/resnet_best_model_Split_1.pth")'''


# test_tests.py (Pi-ready)
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import pandas as pd
import new_import                       # <-- the file above
import test_data as test                # your model-testing utilities
from paths_config import CYTO_DIR, MODEL_CHECKPOINTS

import argparse
from paths_config import MODEL_CHECKPOINTS

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = args.model_name

# Hybrids and CytoFM don’t need an external model checkpoint here.
# If it exists, use it; otherwise, leave it as None.
MODEL_PATH = str(MODEL_CHECKPOINTS[MODEL_NAME]) if MODEL_NAME in MODEL_CHECKPOINTS else None

# 1) Load
df = new_import.import_image_data(str(CYTO_DIR))
print("Total images found:", len(df))
print(df.head(3))

# 2) Split by magnification (your Pi set is 10X-only; x40 may be empty)
x10_df, x40_df = new_import.extract_mag(df)
print("10X:", len(x10_df), "40X:", len(x40_df))
work_df = x10_df if len(x10_df) > 0 else df

print("Using all images for testing:", len(work_df))

# 4) Run the model on the test split (batch_size=1 is safest on Pi)
results = test.test_model(
    work_df,
    model_name=MODEL_NAME,
    model_path=MODEL_PATH,
    batch_size=1,
    device=None
)



print("Done. Metrics (summary):")
for k, v in results.items():
    if k in ("predictions", "labels"):
        try:
            print(f"  {k}: len={len(v)}")
        except Exception:
            print(f"  {k}: (omitted)")
    else:
        print(f"  {k}: {v}")