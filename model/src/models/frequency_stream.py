import torch
import torch.nn as nn

class HaarFilter(nn.Module):
    def __init__(self, in_channels=3):
        super(HaarFilter, self).__init__()
        
        # Haar Wavelet Kernels for LL, LH, HL, HH
        ll = torch.tensor([[1, 1], [1, 1]]) / 2.0
        lh = torch.tensor([[-1, -1], [1, 1]]) / 2.0
        hl = torch.tensor([[-1, 1], [-1, 1]]) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]]) / 2.0
        
        # Process on Grayscale
        self.register_buffer('ll', ll.view(1, 1, 2, 2))
        self.register_buffer('lh', lh.view(1, 1, 2, 2))
        self.register_buffer('hl', hl.view(1, 1, 2, 2))
        self.register_buffer('hh', hh.view(1, 1, 2, 2))
        
    def rgb_to_grayscale(self, x):
        # x is [B, 3, H, W]
        # weights from standard Rec.601
        gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        return gray

    def forward(self, x):
        gray = self.rgb_to_grayscale(x)
        # Apply 2x2 stride 2 conv without bias based on filters
        
        out_ll = nn.functional.conv2d(gray, self.ll, stride=2)
        out_lh = nn.functional.conv2d(gray, self.lh, stride=2)
        out_hl = nn.functional.conv2d(gray, self.hl, stride=2)
        out_hh = nn.functional.conv2d(gray, self.hh, stride=2)
        
        # We care mostly about high-frequency components for artifacts
        # Return a concatenated feature map of the high frequency info stacked
        stacked = torch.cat([out_lh, out_hl, out_hh], dim=1) # Shape: [B, 3, H/2, W/2]
        return stacked

class FrequencyStream(nn.Module):
    def __init__(self):
        """
        Extracts frequency domain features via Haar Wavelet transformations to find structural blending anomalies.
        """
        super(FrequencyStream, self).__init__()
        self.haar = HaarFilter()
        
        # Extract features from the high-frequency maps
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # -> H/4
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # -> H/8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.feature_dim = 128

    def forward(self, x):
        # x shape: [B, 3, H, W]
        freq_maps = self.haar(x)
        features = self.extractor(freq_maps)
        # flatten
        features = torch.flatten(features, 1)
        return features
