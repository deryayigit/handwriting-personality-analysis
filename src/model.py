import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def freeze_backbone(model: ViTClassifier, freeze: bool = True) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = not freeze
