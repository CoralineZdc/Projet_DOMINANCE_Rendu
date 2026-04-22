import torch.nn as nn
from efficientnet_pytorch import EfficientNet  # À installer : pip install efficientnet_pytorch

class EfficientNetVAD(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(EfficientNetVAD, self).__init__()
        # Charge EfficientNet-B0 sans les couches de classification finales
        self.backbone = EfficientNet.from_name('efficientnet-b0', in_channels=3)
        
        # Remplace la tête de classification par une tête de régression VAD
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity() # Enlève la couche FC originale
        
        # Nouvelle tête de régression
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 128),
            nn.SiLU(), # Activation cohérente avec EfficientNet
            nn.Linear(128, 3) # Sortie : Valence, Arousal, Dominance
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regression_head(features)