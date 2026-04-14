# Smart Microscope AI Inference System

## Overview

This project runs AI-based cytology classification on a Raspberry Pi.

The system:

* Captures microscope images
* Runs deep learning inference (ResNet, EfficientNetV2, WaveMix, Hybrid, CytoFM)
* Outputs **Benign (B)** or **Suspicious (Malignant)**
* Displays results in a touchscreen GUI
* Saves predictions to CSV locally (no internet required)

---

## System Architecture

Laptop (Development)
→ GitHub (Source of truth)
→ Raspberry Pi (Deployment)
→ Microscope (Hardware)

---

## Models Supported

* `resnet`  (currently used in GUI)
* `efficientnetv2`
* `wavemix`
* `hybrid`
* `cytofm` (advanced / experimental)

---

## Folder Structure (on Raspberry Pi)

```
~/pi_tests/
└── smart_microscope/
    ├── appdevtest.py
    ├── live_inference.py
    ├── live_smoke_test.py
    ├── test_tests.py
    ├── test_data.py
    ├── ML_models.py
    ├── paths_config.py
    ├── microfocus.py
    ├── models/
    ├── CytoLabeled/
    └── outputs/
```

---

## Setup on Raspberry Pi

### 1. Create environment

```
cd ~/pi_tests
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Install GPIO (required for hardware)

```
sudo apt install -y python3-rpi.gpio
```

---

## Running the System

---

### GUI (Main Application)

Run inside **VNC desktop terminal**:

```
cd ~/pi_tests
source .venv/bin/activate
python smart_microscope/appdevtest.py
```

Login:

```
Username: admin
Password: 1234
```

---

### GUI Features

* Open Camera (requires hardware)
* Capture → runs inference → saves CSV
* Open Image → runs inference (no camera required)
* Manual focus controls (GPIO)
* CSV logging per session

---

## CSV Output

Saved automatically to:

```
~/pi_tests/smart_microscope/outputs/slide_YYYYMMDD_HHMMSS/live_predictions.csv
```

Each row contains:

* timestamp
* image_path
* model_name
* predicted_class
* malignant_probability
* status

---

## Single Image Test (no GUI)

```
python smart_microscope/live_smoke_test.py "smart_microscope/CytoLabeled/B/Image 835.jpeg" resnet
```

---

## Dataset Test

```
PYTHONPATH="$HOME/pi_tests" ./run_and_log.sh smart_microscope/test_tests.py --model_name resnet
```

---

## Model Weights (IMPORTANT)

Do NOT store in GitHub.

Place manually on Pi:

```
~/pi_tests/smart_microscope/models/resnet_best_model_Split_1.pth
```

---

## GitHub Rules

### Included:

* Python code
* Scripts
* Documentation

### Excluded:

* `.venv/`
* `outputs/`
* `logs/`
* model weights (`.pth`, `.pt`)
* captured images

---

## Development Workflow

1. Edit code on laptop
2. Push to GitHub
3. Pull onto Raspberry Pi
4. Run tests

---

## Current Status

Model inference working
GUI working
CSV logging working
VNC desktop working
Hardware integration (camera + microscope) in progress

---

## Notes

* System works fully offline
* WiFi only needed for:

  * SSH
  * file transfer
  * GitHub sync

---

## Author

Anita Itoro Bassey
Smart Microscope AI Project
