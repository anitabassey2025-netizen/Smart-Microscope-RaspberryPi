'''import ML_models
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import os
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from image_path import ImagePathDataset
from torch.utils.data import DataLoader


# MAKES GRADCAM OPTIONAL
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAVE_GRADCAM = True
except Exception:
    HAVE_GRADCAM = False



def load_test_loader(test_data, transform, batch_size=64):

    
    print(f"[INFO] Building testing dataset and dataloader...")
    test_dataset = ImagePathDataset(test_data, transform=transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"[INFO] testing dataset size: {len(test_dataset)}")
    return test_loader


def get_target_layer(model,model_name):

    print(f"Attempting to get target layer of {model_name} model")

    if model_name is None:
        raise ValueError("No model name provided")

    match model_name.lower():
        case "resnet":
            target_layer = model.layer4[-1].conv2

        case "efficientnetv2":
            target_layer = model.features[-1]
        
        case "hybrid":
           
            target_layer = model.projection

        case "wavemix":
            target_layer = model.model.layers[-1]

        case _:
            raise ValueError(f"Unknown model name: {model_name}")
        
    return target_layer



def generate_gradcam(model, input_tensor, model_name, original_size=None, target_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    # Get the correct target layer
    target_layer = get_target_layer(model, model_name)

    # Forward pass to get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = int(torch.argmax(outputs, dim=1).item())

    if target_class is None:
        target_class = predicted_class

    # Setup Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]

    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]  # (H,W)

    # Convert tensor to numpy image for overlay
    input_image_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()  
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    input_image_np = (input_image_np * std) + mean 
    input_image_np = np.clip(input_image_np, 0, 1)

    cam_image = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)  

    print(f"[DEBUG] cam_image shape before resize: {cam_image.shape}")  # ADD THIS
    print(f"[DEBUG] original_size: {original_size}")  # ADD THIS

    if original_size is not None:
        cam_image = np.array(Image.fromarray(cam_image).resize(original_size, Image.LANCZOS))
        print(f"[DEBUG] cam_image shape after resize: {cam_image.shape}")  # ADD THIS

    return cam_image, grayscale_cam, predicted_class

def save_gradcam_image(model_name,cam_image, split_num, label, save_dir="gradcam_outputs", filename="gradcam_example.png"):

    # Create folder if it doesn't exist
    save_dir = save_dir+"_"+model_name
    print(save_dir)
    
    # label dir
    split_dir = os.path.join(save_dir, split_num)
    
    print(split_dir)

    os.makedirs(split_dir, exist_ok=True)

    if(label == 0):
        label_str = "B"

    else:

        label_str = "M"

    label_dir = os.path.join(split_dir, label_str)
    print(label_dir)

    os.makedirs(label_dir, exist_ok=True)

    save_path = os.path.join(label_dir, filename)


    # Save image
    Image.fromarray(cam_image).save(save_path)

    print(f"[INFO] Saved Grad-CAM to {label_dir}")

from paths_config import MODEL_CHECKPOINTS

def test_model(test_df, model_name="resnet", model_path=None, batch_size=64, device=None):
    if model_path is None:
        model_path = str(MODEL_CHECKPOINTS[model_name])

    print(f"Model Path is {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")


    # Load model
    model = ML_models.call_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loader
    test_loader = load_test_loader(test_df, transform, batch_size=batch_size)

    all_preds = []
    all_labels = []
    all_probs = []

    print("[INFO] Running inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            
            # Binary classification - get probability of positive class
            if logits.shape[1] == 1:  # Single output neuron
                probs = torch.sigmoid(logits).squeeze().cpu()
            else:  # Two output neurons [class_0, class_1]
                probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            
            # Get predictions using 0.5 threshold
            preds = (probs >= 0.5).long()
            
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    # Generate GradCAM for every image in the test set
    print("[INFO] Generating Grad-CAMs...")
    
    # Prepare save directory from model path
    base_name = os.path.basename(model_path)         # e.g., "best_model_Split_1.pth"
    split_num = os.path.splitext(base_name)[0].split("best_model_")[-1]  # Extract split number
    
    save_dir = "gradcam_outputs"
    os.makedirs(save_dir, exist_ok=True)

    global_idx = 0  # Track position in dataframe
    
    for images, labels in test_loader:
        images = images.to(device)
        
        for j in range(images.size(0)):
            # Get data from DataFrame for this specific image
            row = test_df.iloc[global_idx]
            image_path = row["image_path"]
            filename = os.path.basename(image_path)
            
            # Get original size from DataFrame
            original_size = (int(row["width"]), int(row["height"]))  # (width, height)
            
            # Generate Grad-CAM
            cam_image, grayscale_cam, pred_class = generate_gradcam(
                model, 
                images[j].unsqueeze(0), 
                model_name=model_name,
                original_size=original_size
            )
            
            print(f"[INFO] Processing {filename} - Original source: {image_path}")

            # Save Grad-CAM
            save_gradcam_image(
                model_name=model_name,
                cam_image=cam_image,
                save_dir=save_dir,
                split_num=split_num,
                label=labels[j].item(),
                filename=filename
            )
            
            global_idx += 1

    # Concatenate all predictions, labels, and probabilities
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Confusion Matrix - FIXED: use all_preds instead of all_labels
    cm = confusion_matrix(all_labels, all_preds)

    # Specificity: TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n[RESULTS]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Confusion Matrix:\n{cm}")

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
    }'''


# test_data.py (Pi-ready: Grad-CAM optional, CPU-safe, robust metrics)

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import ML_models
from image_path import ImagePathDataset
from paths_config import MODEL_CHECKPOINTS, OUTPUTS_DIR

# ---------------------------------------------------------------------
# Grad-CAM (optional) controls
# ---------------------------------------------------------------------
USE_GRADCAM = os.environ.get("USE_GRADCAM", "0") == "1"
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAVE_GRADCAM = True
except Exception:
    HAVE_GRADCAM = False


def load_test_loader(test_data, transform, batch_size=1, num_workers=0):
    """
    Builds a lightweight DataLoader from a DataFrame of image paths + labels.
    Defaults to batch_size=1 for Pi safety; increase if you see plenty of headroom.
    """
    print(f"[INFO] Building testing dataset and dataloader...")
    test_dataset = ImagePathDataset(test_data, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,   # keep 0 on Pi to avoid extra overhead
        pin_memory=False
    )
    print(f"[INFO] testing dataset size: {len(test_dataset)}")
    return test_loader


def get_target_layer(model, model_name):
    """
    Selects the layer to visualize with Grad-CAM based on your model.
    """
    print(f"[INFO] Selecting target layer for {model_name}")
    if model_name is None:
        raise ValueError("No model name provided")

    name = model_name.lower()
    if name == "resnet":
        return model.layer4[-1].conv2
    elif name == "efficientnetv2":
        return model.features[-1]
    elif name == "hybrid":
        return model.projection
    elif name == "wavemix":
        return model.model.layers[-1]
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def generate_gradcam(model, input_tensor, model_name, original_size=None, target_class=None):
    """
    Returns (cam_image_uint8, grayscale_cam_float, predicted_class_int).
    Requires pytorch-grad-cam to be installed. Caller should guard with HAVE_GRADCAM/USE_GRADCAM.
    """
    device = next(model.parameters()).device
    model.eval()
    input_tensor = input_tensor.to(device)

    target_layer = get_target_layer(model, model_name)

    # Predict class first
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = int(torch.argmax(outputs, dim=1).item())

    if target_class is None:
        target_class = predicted_class

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]

    # (1,1,H,W) -> [0] (H,W)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # De-normalize input for overlay (assumes ImageNet stats in transforms)
    # If you changed normalize stats, mirror those here.
    img_np = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)  # uint8

    if original_size is not None:
        try:
            cam_image = np.array(Image.fromarray(cam_image).resize(original_size, Image.LANCZOS))
        except Exception:
            pass  # if resize fails, keep CAM as-is

    return cam_image, grayscale_cam, predicted_class


def save_gradcam_image(model_name, cam_image, split_num, label, filename, save_root=OUTPUTS_DIR):
    """
    Saves CAM to: OUTPUTS_DIR/gradcam_<model_name>/<split>/<B|M>/<filename>
    """
    subdir = save_root / f"gradcam_{model_name}" / split_num / ("B" if int(label) == 0 else "M")
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / filename

    Image.fromarray(cam_image).save(out_path)
    print(f"[INFO] Saved Grad-CAM -> {out_path}")


def _compute_specificity(cm: np.ndarray) -> float:
    """
    Specificity = TN / (TN + FP)
    Handle edge cases where only one class appears.
    """
    # cm can be 1x1 (only one class present) or 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        denom = (tn + fp)
        return float(tn) / denom if denom > 0 else 0.0
    # Fallback: no negatives to measure
    return 0.0


def test_model(test_df, model_name="resnet", model_path=None, batch_size=1, device=None):
    """
    Runs inference over test_df on the Pi.

    - model_name: one of {"resnet","efficientnetv2","wavemix","cytofm",
                          "hybrid_cytofm_resnet","hybrid_cytofm_efficientnetv2","hybrid_cytofm_wavemix"}
    - model_path: path to .pth; if None, taken from MODEL_CHECKPOINTS (only for plain CNNs)
    - batch_size: default 1 on Pi for safety (forced to 1 for adapters)
    """
    import os
    import numpy as np
    import torch
    from torchvision import transforms
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # -------- model family flags --------
    name = (model_name or "").lower()
    is_adapter = (name == "cytofm") or name.startswith("hybrid_cytofm_")   # CytoFM + all hybrids
    is_plain_cnn = not is_adapter                                          # resnet/efficientnetv2/wavemix

    # -------- resolve checkpoint & device --------
    if model_path is None:
        # For adapters, it's OK to have no checkpoint here; their engines load internally.
        # For plain CNNs, fall back to MODEL_CHECKPOINTS.
        model_path = str(MODEL_CHECKPOINTS[model_name]) if (not is_adapter and model_name in MODEL_CHECKPOINTS) else None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Model Path: {model_path}")

    # Plain CNNs require a checkpoint; adapters do not.
    if is_plain_cnn:
        if model_path is None or (isinstance(model_path, str) and not os.path.exists(model_path)):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # -------- build model (+weights if needed) --------
    model = ML_models.call_model(model_name)

    # Only load external weights into plain CNNs. Adapters load internally.
    if is_plain_cnn:
        state = torch.load(model_path, map_location=device)
        # If your .pth is nested, unwrap here:
        # state = state.get("model_state_dict", state)
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    # -------- transforms --------
    if is_adapter:
        # Adapter models (CytoFM + all hybrids): let engine do tiling/normalization.
        transform = transforms.Compose([
            transforms.ToTensor(),   # HWC uint8 -> CHW float in [0,1]
        ])
        if batch_size != 1:
            print("[WARN] For adapter models, forcing batch_size=1 for CPU stability.")
        batch_size = 1
    else:
        # Plain CNNs: ImageNet preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    # -------- dataloader --------
    test_loader = load_test_loader(test_df, transform, batch_size=batch_size)

    # -------- inference loop (binary) --------
    all_preds, all_labels, all_probs = [], [], []

    print("[INFO] Running inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Works for both CNNs and adapters (adapters return logits as well)
            logits = model(images)

            # Binary probability of positive class
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze(dim=1).detach().cpu()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()

            preds = (probs >= 0.5).long()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())
            all_probs.append(probs)

    # -------- concat (guard empty) --------
    all_preds = torch.cat(all_preds) if len(all_preds) else torch.empty(0, dtype=torch.long)
    all_labels = torch.cat(all_labels) if len(all_labels) else torch.empty(0, dtype=torch.long)
    all_probs = torch.cat(all_probs) if len(all_probs) else torch.empty(0, dtype=torch.float32)

    # -------- Grad-CAM (plain CNNs only) --------
    use_cam = (USE_GRADCAM and HAVE_GRADCAM and is_plain_cnn)
    if use_cam and model_path:
        print("[INFO] Generating Grad-CAMs...")
        base_name = os.path.basename(model_path)  # e.g., "resnet_best_model_Split_1.pth"
        split_num = os.path.splitext(base_name)[0]
        if "best_model_" in split_num:
            split_num = split_num.split("best_model_")[-1]

        import pandas as pd
        global_idx = 0
        for images, labels in test_loader:
            images = images.to(device)
            for j in range(images.size(0)):
                row = test_df.iloc[global_idx]
                image_path = row.get("image_path", None)
                filename = os.path.basename(image_path) if image_path else f"img_{global_idx}.png"

                if "width" in row and "height" in row and not (pd.isna(row["width"]) or pd.isna(row["height"])):
                    original_size = (int(row["width"]), int(row["height"]))  # (W, H)
                else:
                    original_size = None

                try:
                    cam_image, grayscale_cam, pred_class = generate_gradcam(
                        model,
                        images[j].unsqueeze(0),
                        model_name=model_name,
                        original_size=original_size
                    )
                    save_gradcam_image(
                        model_name=model_name,
                        cam_image=cam_image,
                        split_num=split_num,
                        label=labels[j].item(),
                        filename=filename,
                        save_root=OUTPUTS_DIR
                    )
                except Exception as e:
                    print(f"[WARN] Grad-CAM failed for {filename}: {e}")
                finally:
                    global_idx += 1
    else:
        if USE_GRADCAM and not HAVE_GRADCAM and is_plain_cnn:
            print("[INFO] pytorch-grad-cam not installed; skipping CAM.")
        else:
            print("[INFO] Grad-CAM disabled (or adapter model).")

    # -------- metrics --------
    y_true = all_labels.numpy() if all_labels.numel() else []
    y_pred = all_preds.numpy() if all_preds.numel() else []

    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) if len(y_true) else np.zeros((2, 2), dtype=int)
    specificity = _compute_specificity(cm) if cm.sum() > 0 else 0.0

    print("\n[RESULTS]")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Confusion Matrix:\n{cm}")

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