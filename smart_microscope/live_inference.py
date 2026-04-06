from pathlib import Path
from datetime import datetime
import csv

import cv2
import torch
from PIL import Image
from torchvision import transforms

import ML_models
from paths_config import MODEL_CHECKPOINTS, OUTPUTS_DIR


class LivePredictor:
    def __init__(self, model_name="resnet", threshold=0.5, device=None):
        self.model_name = model_name.lower()
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_name not in MODEL_CHECKPOINTS:
            raise ValueError(f"Unknown model name: {self.model_name}")

        self.model_path = Path(MODEL_CHECKPOINTS[self.model_name])
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        print(f"[LIVE] Loading model: {self.model_name}")
        print(f"[LIVE] Checkpoint: {self.model_path}")
        print(f"[LIVE] Device: {self.device}")

        self.model = ML_models.call_model(self.model_name)

        state = torch.load(self.model_path, map_location=self.device)
        # If later your saved checkpoint is nested, use:
        # state = state.get("model_state_dict", state)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def predict_image(self, image_path):
        image_path = Path(image_path)

        with Image.open(image_path).convert("RGB") as img:
            x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

            if logits.ndim != 2:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

            if logits.shape[1] == 1:
                prob_m = float(torch.sigmoid(logits).item())
            else:
                prob_m = float(torch.softmax(logits, dim=1)[0, 1].item())

        pred_class = 1 if prob_m >= self.threshold else 0
        status = "Suspicious" if pred_class == 1 else "Not suspicious"

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(image_path),
            "image_name": image_path.name,
            "model_name": self.model_name,
            "predicted_class": pred_class,
            "malignant_probability": prob_m,
            "status": status,
        }

    def append_to_csv(self, result, csv_path):
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "image_path",
                    "image_name",
                    "model_name",
                    "predicted_class",
                    "malignant_probability",
                    "status",
                ])
            writer.writerow([
                result["timestamp"],
                result["image_path"],
                result["image_name"],
                result["model_name"],
                result["predicted_class"],
                f'{result["malignant_probability"]:.6f}',
                result["status"],
            ])

    def draw_overlay(self, frame_bgr, result):
        overlay = frame_bgr.copy()
        text = f'{result["status"]} | P(M)={result["malignant_probability"]:.2f}'
        color = (0, 0, 255) if result["predicted_class"] == 1 else (0, 255, 0)

        cv2.putText(
            overlay,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )
        return overlay
