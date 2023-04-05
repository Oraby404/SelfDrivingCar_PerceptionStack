import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# Encoder (Down-sampling Block)
# The encoder is a stack of various conv blocks
class DownSamplingBlock(nn.Module):
    def __init__(self, inChannels, n_filters, dropout_prob=0.0, max_pooling=True):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling

        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)

        if dropout_prob > 0.0:
            self.drop = nn.Dropout(p=dropout_prob)

        if max_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        conv = F.relu(self.bn1(self.conv1(x)))

        conv = F.relu(self.bn2(self.conv2(conv)))

        if self.dropout_prob > 0.0:
            conv = self.drop(conv)

        if self.max_pooling:
            next_layer = self.pool(conv)

        else:
            next_layer = conv

        skip_connection = conv

        return next_layer, skip_connection


# Decoder (Up-sampling Block)
# Takes the arguments expansive_input (which is the input tensor from the previous layer)
# and contractive_input (the input tensor from the previous skip layer)
class UpSamplingBlock(nn.Module):
    def __init__(self, inChannels, n_filters):
        super().__init__()

        # out = (in-1)*s -2p + d(k-1) + op + 1
        self.upSample = nn.ConvTranspose2d(in_channels=inChannels, out_channels=n_filters, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, expansive_input, contractive_input):
        up = self.upSample(expansive_input)

        if up.shape != contractive_input.shape:
            # crop the contractive input to fit the Up sampled expansive input
            (B, C, H, W) = up.shape
            contractive_input = transforms.CenterCrop([H, W])(contractive_input)

        merge = torch.cat([up, contractive_input], dim=1)

        conv = F.relu(self.bn1(self.conv1(merge)))

        conv = F.relu(self.bn2(self.conv2(conv)))

        return conv


class UNet(nn.Module):
    def __init__(self, inChannels=3, n_filters=64, n_classes=13):
        super().__init__()

        encChannels = (inChannels, n_filters, 128, 256, 512, 1024)
        decChannels = (1024, 512, 256, 128, n_filters)

        self.encBlocks = nn.ModuleList([
            DownSamplingBlock(encChannels[i], encChannels[i + 1]) for i in range(len(encChannels) - 2)
        ])
        self.lastEncBlocks = DownSamplingBlock(encChannels[len(encChannels) - 2], encChannels[len(encChannels) - 1],
                                               dropout_prob=0.3, max_pooling=False)

        self.decBlocks = nn.ModuleList([
            UpSamplingBlock(decChannels[i], decChannels[i + 1]) for i in range(len(decChannels) - 1)
        ])

        self.conv = nn.Conv2d(in_channels=n_filters, out_channels=n_classes, kernel_size=1, stride=1)

    def forward(self, x):

        skipConnections = []

        # Contracting Path (encoding)
        next_layer = x
        for enc_block in self.encBlocks:
            next_layer, skip_connection = enc_block(next_layer)
            skipConnections.append(skip_connection)

        next_layer, _ = self.lastEncBlocks(next_layer)

        # Expanding Path (decoding)
        for dec_block in self.decBlocks:
            next_layer = dec_block(next_layer, skipConnections.pop())

        conv = self.conv(next_layer)

        return conv

