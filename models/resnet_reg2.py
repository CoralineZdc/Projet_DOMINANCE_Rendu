import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class RegressionHead(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0, hidden_features=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetRegression(nn.Module):
    def __init__(self, block, num_blocks, num_outputs=3, dropout_rate=0.0, separate_heads=False):
        super(ResNetRegression, self).__init__()
        self.in_planes = 64
        self.separate_heads = separate_heads

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        out_features = 512 * block.expansion
        if separate_heads:
            self.head_valence = RegressionHead(out_features, 1, dropout_rate=dropout_rate)
            self.head_arousal = RegressionHead(out_features, 1, dropout_rate=dropout_rate)
            self.head_dominance = RegressionHead(out_features, 1, dropout_rate=dropout_rate)
        else:
            self.linear = RegressionHead(out_features, num_outputs, dropout_rate=dropout_rate)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)

        if self.separate_heads:
            val = self.head_valence(out)
            aro = self.head_arousal(out)
            dom = self.head_dominance(out)
            return torch.cat([val, aro, dom], dim=1)
        return self.linear(out)


def ResNet18RegressionThreeOutputs(dropout_rate=0.0, separate_heads=False):
    return ResNetRegression(
        BasicBlock,
        [2, 2, 2, 2],
        num_outputs=3,
        dropout_rate=dropout_rate,
        separate_heads=separate_heads,
    )


class ResNet50PretrainedRegressionModel(nn.Module):
    def __init__(self, dropout_rate=0.0, separate_heads=False, freeze_backbone=False):
        super(ResNet50PretrainedRegressionModel, self).__init__()
        self.backbone = tv_models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.separate_heads = separate_heads

        if separate_heads:
            self.head_valence = RegressionHead(2048, 1, dropout_rate=dropout_rate, hidden_features=512)
            self.head_arousal = RegressionHead(2048, 1, dropout_rate=dropout_rate, hidden_features=512)
            self.head_dominance = RegressionHead(2048, 1, dropout_rate=dropout_rate, hidden_features=512)
        else:
            self.linear = RegressionHead(2048, 3, dropout_rate=dropout_rate, hidden_features=512)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)

        if self.separate_heads:
            val = self.head_valence(features)
            aro = self.head_arousal(features)
            dom = self.head_dominance(features)
            return torch.cat([val, aro, dom], dim=1)
        return self.linear(features)


def ResNet50PretrainedRegressionThreeOutputs(dropout_rate=0.0, separate_heads=False, freeze_backbone=False):
    return ResNet50PretrainedRegressionModel(
        dropout_rate=dropout_rate,
        separate_heads=separate_heads,
        freeze_backbone=freeze_backbone,
    )


def ResNet50RegressionThreeOutputs(dropout_rate=0.0, separate_heads=False):
    return ResNetRegression(
        Bottleneck,
        [3, 4, 6, 3],
        num_outputs=3,
        dropout_rate=dropout_rate,
        separate_heads=separate_heads,
    )


def ResNet18RegressionTwoOutputs():
    # Backward-compatible alias
    return ResNetRegression(BasicBlock, [2, 2, 2, 2], num_outputs=1)
