#defines the models being called
import torch
import torch.nn as nn
import torchvision.models as models
from wavemix.classification import WaveMix
#from torchvision.models import EfficientNet_V2_S_Weights
try:
    from wavemix.classification import WaveMix  # some versions layout this way
except ModuleNotFoundError:
    from wavemix import WaveMix  # others expose WaveMix at the package root

model_names = ["resnet","efficientnetv2","wavemix","hybrid","cytofm"]

#implementation derived from https://pypi.org/project/wavemix/ spefically image classification
class WaveMixClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = WaveMix(
            num_classes= num_classes, 
            depth= 16,
            mult= 2,
            ff_channel= 192,
            final_dim= 192,
            dropout= 0.5,
            level=3,
            patch_size=4,
        )
        
        
    def forward(self, x):
        return self.model(x)



class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=2, cnn_pretrained=False, embed_dim=256, num_heads=8, depth=6):
        super().__init__()

        # --- CNN Backbone (ResNet18) ---
        resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # output: (B,512,7,7)

        self.projection = nn.Conv2d(512, embed_dim, kernel_size=1)  # (B, embed_dim, 7,7)

        self.patch_dim = 7 * 7  # number of patches = 49

        # flatten spatial dims
        self.flatten = nn.Flatten(2)  # (B, embed_dim, 49)

        # transpose to Transformer format
        self.transpose = lambda x: x.transpose(1,2)  # (B,49,embed_dim)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = self.cnn(x)                 # (B,512,7,7)
        x = self.projection(x)          # (B,embed_dim,7,7)
        x = self.flatten(x)             # (B,embed_dim,49)
        x = self.transpose(x)           # (B,49,embed_dim)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) 
        x = torch.cat([cls, x], dim=1)  # (B,50,embed_dim)

        x = self.transformer(x)         # (B,50,embed_dim)

        cls_out = x[:, 0]               # take CLS
        return self.fc(cls_out)

# === CytoFM adapter as nn.Module (no changes to the existing harness) ===
# === CytoFM adapter as nn.Module (no changes to existing harness) ===
class CytoFMAdapter(nn.Module):
    """
    Wraps the CytoFM+ABMIL inference engine so it behaves like an nn.Module.
    forward(x) -> logits (B,2)
    """
    def __init__(self):
        super().__init__()

        # paths_config import: try absolute (package), then relative (module)
        try:
            from smart_microscope import paths_config as paths
        except ModuleNotFoundError:
            import paths_config as paths

        # engine import: try absolute (package), then relative (module)
        try:
            from smart_microscope.cytofm.cytofm_infer import CytoFMABMILInference
        except ModuleNotFoundError:
            from cytofm.cytofm_infer import CytoFMABMILInference

        self.engine = CytoFMABMILInference(
            cytofm_weights=str(paths.CYTOFM_BACKBONE),
            abmil_weights=str(paths.CYTOFM_HEAD),
            device="cpu",
            patch_size=256,
            stride=256
        )

    def forward(self, x):
        import numpy as np, cv2, torch
        B = x.shape[0]
        logits = []
        for i in range(B):
            # Convert torch RGB [3,H,W] → uint8 BGR [H,W,3] for the engine
            img = x[i].detach().cpu().numpy()              # [3,H,W], float
            img = np.transpose(img, (1,2,0))               # [H,W,3], RGB
            img = (img * 255.0).clip(0,255).astype(np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            out = self.engine.predict(bgr)                 # {"probs":[p_B,p_M], ...}
            p = out["probs"]
            # convert probs → logits so your metrics expect the same type
            eps = 1e-6
            p = np.clip(p, eps, 1. - eps)
            logit = np.log(p / (1. - p))                   # shape (2,)
            logits.append(torch.tensor(logit, dtype=torch.float32))
        return torch.stack(logits, dim=0)                  # [B,2]


def get_CytoFM_Adapter():
    print("Calling CytoFM (Adapter as nn.Module; engine loads backbone+ABMIL).")
    return CytoFMAdapter()

def get_efficient(num_classes = 2):

    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    print(f"Calling EfficientnetV2-s with {num_classes} number of classes")

    return model

def get_Res(num_classes = 2):

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features,num_classes)

    print(f"Calling ResNet18 with {num_classes} number of classes")

    return model

def get_Hybrid(num_classes=2):
    print(f"Calling Hybrid with {num_classes} number of classes")
    return HybridCNNViT(num_classes=num_classes)

def get_Wavemix(num_classes=2):
    print(f"Calling Wavemix with {num_classes} number of classes")
    return WaveMixClassifier(num_classes=num_classes)

def get_Hybrid_CytoFM_ResNet():
    print("Calling Hybrid (CytoFM + ResNet18)")
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
        tile_bs=4 #change back to 8
    )

def get_Hybrid_CytoFM_EfficientNetV2():
    print("Calling Hybrid (CytoFM + EfficientNetV2-S)")
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
        tile_bs=2
    )

def get_Hybrid_CytoFM_WaveMix():
    print("Calling Hybrid (CytoFM + WaveMix)")
    from smart_microscope import paths_config as paths
    from smart_microscope.hybrid.hybrid_infer import HybridCytoFMAdapter
    fusion = paths.ROOT.parent / "models" / "fusion_head_wavemix.pt"
    return HybridCytoFMAdapter(
        cytofm_weights=str(paths.CYTOFM_BACKBONE),
        abmil_weights=str(paths.CYTOFM_HEAD),
        fusion_head=str(fusion),
        cnn_backbone="wavemix",
        cnn_embed_dim=192,
        device="cpu",
        tile_bs=2
    )

def call_model(model_name=None):
    print(f"Attempting to fetch {model_name} model")

    if model_name is None:
        raise ValueError("No model name provided")

    match model_name.lower():
        case "resnet":
            return get_Res()
        case "efficientnetv2":
            return get_efficient()
        case "hybrid":
            return get_Hybrid()
        case "wavemix":
            return get_Wavemix()
        case "cytofm":                         # <-- add this one new case
            return get_CytoFM_Adapter()
        case "hybrid_cytofm_resnet":
            return get_Hybrid_CytoFM_ResNet()
        case "hybrid_cytofm_efficientnetv2":
            return get_Hybrid_CytoFM_EfficientNetV2()
        case "hybrid_cytofm_wavemix":
            return get_Hybrid_CytoFM_WaveMix()
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
        





