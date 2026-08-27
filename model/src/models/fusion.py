import torch
import torch.nn as nn
from .spatial_stream import SpatialStream
from .frequency_stream import FrequencyStream

class MultiStreamDetector(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(MultiStreamDetector, self).__init__()
        
        self.spatial = SpatialStream(model_name='tf_efficientnet_b4_ns', pretrained=True, freeze_backbone=freeze_backbone)
        self.frequency = FrequencyStream()
        
        combined_dim = self.spatial.feature_dim + self.frequency.feature_dim
        
        # Aggregation Head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        f_spat = self.spatial(x)
        f_freq = self.frequency(x)
        
        # Concatenate multi-modal features
        f_fused = torch.cat([f_spat, f_freq], dim=1)
        
        logits = self.classifier(f_fused)
        return logits
