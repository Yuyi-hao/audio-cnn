import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # batch normalization 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) # batch normalization 1
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x, feature_map_dict=None, prefix=""):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out+shortcut
        if feature_map_dict is not None:
            feature_map_dict[f"{prefix}.conv"] = out_add
        out = torch.relu(out+shortcut)
        if feature_map_dict is not None:
            feature_map_dict[f"{prefix}.relu"] = out
        return out



    
class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # residual layers
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for _ in range(3)])
        self.layer2 = nn.ModuleList([ResidualBlock(64, 128, stride=2)]+[ResidualBlock(128, 128) for _ in range(3)])
        self.layer3 = nn.ModuleList([ResidualBlock(128, 256, stride=2)]+[ResidualBlock(256, 256) for _ in range(5)])
        self.layer4 = nn.ModuleList([ResidualBlock(256, 512, stride=2)]+[ResidualBlock(512, 512) for _ in range(2)])

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x, return_feature_maps=False):
        feature_maps = {}
        x = self.conv1(x)
        feature_maps["conv1"] = x

        for idx, block in enumerate(self.layer1):
            x = block(x, feature_maps, prefix=f"Layer1.block{idx}")
        feature_maps["layer1"] = x
        for idx, block in enumerate(self.layer2):
            x = block(x, feature_maps, prefix=f"Layer2.block{idx}")
        feature_maps["layer2"] = x
        for idx, block in enumerate(self.layer3):
            x = block(x, feature_maps, prefix=f"Layer3.block{idx}")
        feature_maps["layer3"] = x
        for idx, block in enumerate(self.layer4):
            x = block(x, feature_maps, prefix=f"Layer4.block{idx}")
        feature_maps["layer4"] = x

        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        if not return_feature_maps:
            return x
        else:
            return x, feature_maps

