import torch
import torch.nn as nn

# Model architecture definition
class CNN(nn.Module):
    def __init__(self, channels, dropout):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=channels, kernel_size=1, stride = 1, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=9, stride = 1, padding=9//2, groups = channels)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=19, stride = 1, padding=19//2, groups = channels)

        self.bn = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Conv2d(in_channels=3*channels, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.out_nonlinear = nn.ReLU()
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        
    def forward(self, x, mask):
        B, N, C = x.shape

        x = x.transpose(1, 2).view(B, C, N, 1)
        x = self.conv1(x)
        x = self.out_nonlinear(x)
        x = self.bn(x)
        
        # First parallel
        z1 = self.conv2(x)
        z1 = self.out_nonlinear(z1)
        z1 = self.bn(z1)

        # First parallel        
        z2 = self.conv3(x)
        z2 = self.out_nonlinear(z2)
        z2 = self.bn(z2)

        x = torch.cat([x, z1, z2], dim=1)
        x = self.dropout(x)
        x = self.final_conv(x)
        x = x.view(B, N)
        x = x * mask
        return x
