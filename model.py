import torch
import torch.nn as nn
import torch.nn.functional as F
from env import ACTION_SPACE_SIZE

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=5, num_channels=128):
        super(ChessNet, self).__init__()
        self.conv_initial = nn.Conv2d(119, num_channels, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(73)
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy
        p = self.policy_bn(self.policy_conv(x))
        policy_logits = p.flatten(1)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value
