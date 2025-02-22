import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU()
            )
        self.enc1 = conv_block(in_channels, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = conv_block(64, 128)
        self.dec2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1x1 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d2 = self.dec2(e2)
        out = self.conv1x1(d2)
        return out