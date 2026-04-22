import torch
import torch.nn.functional as F
from torch import nn

# --- Constantes adaptées à FER2013 (48x48) ---
# Plus besoin de définir image_w/h fixes, le modèle s'adapte, 
# mais on garde en tête que l'entrée attendue est 48x48.

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class GDConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileFaceNetVAD(nn.Module):
    """
    MobileFaceNet adapté pour la régression VAD sur images 48x48.
    """
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, num_outputs=3, dropout_rate=0.3):
        super(MobileFaceNetVAD, self).__init__()
        block = InvertedResidual
        input_channel = 64
        last_channel = 512

        if inverted_residual_setting is None:
            # Configuration originale
            inverted_residual_setting = [
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]

        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        
        # MODIFICATION 1: stride=1 au lieu de 2 pour préserver la résolution 48x48
        # 48 -> 24 (avec stride 2) est trop agressif dès la première couche pour ce réseau profond.
        # On garde stride=2 ici, mais on surveille la taille. 
        # Calcul: 48/2 = 24. Ensuite les blocks réduisent encore.
        # Pour être sûr, on passe à stride=1 ici, et on laisse les blocks faire la réduction.
        self.conv1 = ConvBNReLU(3, input_channel, stride=1) 
        
        self.dw_conv = DepthwiseSeparableConv(in_planes=64, out_planes=64, kernel_size=3, padding=1)
        
        features = list()
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                # MODIFICATION 2: Réduire le stride des premiers blocks si nécessaire?
                # Non, gardons la structure, mais vérifions la taille finale.
                # Avec entrée 48, stride=1 au debut -> 48.
                # Block 1 (s=2) -> 24.
                # Block 2 (s=2) -> 12.
                # Block 3 (s=1) -> 12.
                # Block 4 (s=2) -> 6.
                # Block 5 (s=1) -> 6.
                # Sortie features: 6x6.
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        self.conv2 = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        
        # MODIFICATION 3: Ajuster le kernel du GDConv.
        # Si la sortie de features est 6x6, un kernel 7 est trop grand.
        # On met kernel_size=6 pour faire un Global Pooling sur 6x6 -> 1x1.
        # On peut aussi utiliser AdaptiveAvgPool, mais GDConv est spécifique à MobileFaceNet.
        # On va définir le kernel dynamiquement ou le fixer à 6 pour du 48x48.
        self.gdconv_kernel_size = 6 
        self.gdconv = GDConv(in_planes=512, out_planes=512, kernel_size=self.gdconv_kernel_size, padding=0)
        
        self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        
        self.features = nn.Sequential(*features)

        # MODIFICATION 4: Tête de régression VAD
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_outputs)
        )

        # Initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.features(x)
        x = self.conv2(x)
        
        # Vérification de sécurité pour la taille du kernel GDConv
        # Si la taille spatiale n'est pas 6, on adapte (optionnel, mais robuste)
        h, w = x.shape[2], x.shape[3]
        if h != self.gdconv_kernel_size:
            # Si la taille change, on recrée le GDConv dynamiquement ou on utilise un pooling
            # Pour simplifier ici, on utilise un AdaptiveAvgPool2d si la taille ne correspond pas
            x = F.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = self.gdconv(x)
            
        x = self.conv3(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1) # Flatten (Batch, 128)
        
        return self.regression_head(x)

# Fonction utilitaire pour instancier le modèle facilement
def MobileFaceNetVAD_Pretrained(dropout_rate=0.3, num_outputs=3):
    return MobileFaceNetVAD(dropout_rate=dropout_rate, num_outputs=num_outputs)

if __name__ == "__main__":
    # Test avec une entrée 48x48
    model = MobileFaceNetVAD()
    dummy_input = torch.randn(4, 3, 48, 48) # Batch de 4, 3 canaux, 48x48
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}") # Doit être torch.Size([4, 3])
        print("Modèle validé avec succès pour 48x48 !")
    except Exception as e:
        print(f"Erreur lors du test: {e}")