import torch
import torch.nn as nn
import timm

class SpatialStream(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True, freeze_backbone=True):
        """
        Extracts spatial (texture and semantic) features from the image using an EfficientNet backbone.
        """
        super(SpatialStream, self).__init__()
        
        # We drop the final classification head (num_classes=0) to use as feature extractor
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

        # Parameter-Efficient Fine-Tuning (PEFT) strategy
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Freeze mostly everything, but potentially fine-tune LayerNorms/BatchNorms if desired
                if 'bn' not in name and 'norm' not in name:
                    param.requires_grad = False

    def forward(self, x):
        # x expected shape: [Batch, 3, 224, 224]
        features = self.backbone(x)
        return features
