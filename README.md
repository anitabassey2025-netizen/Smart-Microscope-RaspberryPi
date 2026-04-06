\# Smart Microscope – Offline Cytology Inference System (Raspberry Pi)



This repository contains the full \*\*source code, documentation, and deployment instructions\*\*

for a Raspberry‑Pi–based smart microscope system developed for \*\*offline cytological analysis\*\*

in low‑resource and rural settings (e.g., mobile clinics in Mali).



The system supports:

\- Standard CNN classifiers (ResNet18, EfficientNetV2‑S, WaveMix)

\- CytoFM (Vision Transformer + ABMIL for tile‑based cytology inference)

\- Hybrid models combining CytoFM global features with CNN local features

\- Fully offline inference once provisioned



> \*\*Important:\*\* Trained model weight files (`.pth`, `.pt`) are \*\*not included\*\* in this repository due to size limits and deployment requirements.  

> See \*\*“Model Weights \& Deployment”\*\* below.



\---



\## System Overview



The Smart Microscope pipeline is designed for \*\*offline‑first medical inference\*\*:



1\. Samples are imaged using a microscope + camera

2\. Images are tiled and processed locally on a Raspberry Pi

3\. CNNs and/or CytoFM models perform inference

4\. Results are logged locally (no cloud dependency during use)



Once the system is provisioned, \*\*no internet connection is required\*\* for clinical operation.



\---



\## Supported Models



\### CNN Baselines

\- ResNet18

\- EfficientNetV2‑S

\- WaveMix



\### CytoFM

\- Vision Transformer (ViT‑B/16)

\- Adaptive Batch Multiple Instance Learning (ABMIL)

\- Tile‑level aggregation for whole‑slide inference



\### Hybrid Models

\- CytoFM + ResNet18

\- CytoFM + EfficientNetV2‑S  

(using lightweight fusion heads)



\---



\## Repository Structure

smart\_microscope/

├── cytofm/                # CytoFM backbone and inference

├── hybrid/                # Hybrid inference adapters

├── models/                # (EMPTY in repo – see Model Weights section)

├── test\_data.py

├── test\_tests.py

├── ML\_models.py

├── image\_path.py

├── live\_inference.py

├── microfocus.py

├── outputs/               # runtime outputs (not tracked)

└── ...

scripts/

└── run\_and\_log.sh

docs/

└── project documentation

README.md

\---



\## Installation (Initial Setup – Internet Required)



These steps are performed \*\*once\*\*, during system provisioning.



```bash

git clone https://github.com/anitabassey2025-netizen/Smart-Microscope-RaspberryPi.git

cd Smart-Microscope-RaspberryPi



python3 -m venv .venv

source .venv/bin/activate



pip install -r requirements.txt





\###Model Weights \& Deployment (IMPORTANT)

This repository does not include trained model weight files (.pth, .pt) due to size limits and offline deployment requirements. During initial SD card setup, while the Raspberry Pi has access to a reliable internet connection (e.g., home Wi‑Fi, university network, or office hotspot), users must download the model weight files from the provided Google Drive link and manually place them into \~/pi\_tests/smart\_microscope/models/. This download step is performed once during installation, before field use. After the weights are saved locally, the system runs fully offline, and no further internet access is required for patient testing in the field.

Before running the system, ensure the following files exist in:



\~/pi\_tests/smart\_microscope/models/



Required:

\- cytofm\_weights.pth (≈1.4 GB)

\- resnet\_best\_model\_Split\_1.pth

\- efficientnetv2\_best\_model\_Split\_1.pth

\- wavemix\_best\_model\_Split\_1.pth

\- fusion\_head\_resnet.pt

\- fusion\_head\_effnet.pt



These files must be placed in the directory above before running inference.





\### Running Inference

1\. Flash OS onto Raspberry Pi

2\. Clone GitHub repo (code only)

3\. Create virtual environment

4\. Download model weights ONCE

5\. Copy weights into \~/pi\_tests/smart\_microscope/models/

6\. Run smoke tests

7\. Seal the system

&#x09;### CODING Step-By-Step Instructions:

&#x09;3.) Create virtual environment
	Activate the environment:

&#x09;cd \~/pi\_tests

&#x09;source .venv/bin/activate

&#x09;export PYTHONPATH="$HOME/pi\_tests"
	6.) Run Smoke tests

&#x09;./run\_and\_log.sh smart\_microscope/test\_tests.py --model\_name resnet

&#x09;./run\_and\_log.sh smart\_microscope/test\_tests.py --model\_name cytofm

&#x09;./run\_and\_log.sh smart\_microscope/test\_tests.py --model\_name hybrid\_cytofm\_resnet





\###Designed for Offline \& Rural Use



No internet required after setup

All computation runs locally on the Raspberry Pi

Suitable for mobile clinics and low‑connectivity environments

Model weights pre‑loaded during provisioning





\###License

This project is provided for research and deployment collaboration purposes.

Contact the project maintainers for usage and deployment permissions.

