import torchvision 
resnet = torchvision.models.resnet.resnet50(pretrained=True)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding = 1, kernel_size = 3, stride = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
                ConvBlock(in_channels, out_channels), 
                ConvBlock(in_channels, out_channels)
            )

    def forward(self, x):
        return self.bridge(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
                nn.Upsample(mode = 'bilinear', scale_factor =2), 
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride =1)
                )

        self.conv_block1 = ConvBlock(in_channels, out_channels)
        self.conv_block2 = ConvBlock(out_channels, out_channels)

    def forward(self, x , d_x):

        x = self.upsample(x)
        x = torch.cat([x, d_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        return x

class UNetResnet(nn.Module):

    def __init__(self, n_class = 1, resnet):
        super().__init__()
        down_blocks = []
        up_blocks = []
        # Why3  ? 
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool =  list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bootleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.dconv = DConv(2048, 2048)
        
        up_blocks.append(UpBlock(2048, 1024))
        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512 ,256))
        up_blocks.append(UpBlock(in_channels = 128 + 64, out_channels = 128))
        up_blocks.append(UpBlock(in_channels = 64 + 3, out_channels = 64))
        
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size = 1, stride = 1)

        
