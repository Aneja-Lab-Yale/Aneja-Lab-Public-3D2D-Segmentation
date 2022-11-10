# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# --------------------------------------------------- Imports -----------------------------------------------------

import torch
from torch import nn
from torch.nn.functional import pad


# ------------------------------------------------- UNet3D model ---------------------------------------------

class UNet3D(nn.Module):
    """
    This class defines the architecture of the 3D UNet
    """

    def __init__(self, Ci=1, Co=1, xpad=True):
        """
        Inputs:
            - Ci: number of input channels into the 3D UNet
            - Co: number of output channels out of the 3D UNet
            - xpad: if the input size is not 2^n in all dimensions, set xpad to True.
        """
        super().__init__()

        # Downsampling limb of UNet:
        self.left1 = Doubleconv(Ci, 64)
        self.left2 = DownDoubleconv(64, 128)
        self.left3 = DownDoubleconv(128, 256)
        self.left4 = DownDoubleconv(256, 512)

        # Bottleneck of UNet:
        self.bottom = DownDoubleconv(512, 1024)

        # Upsampling limb of UNet:
        # the units are numbered in reverse to match the corresponding downsampling units
        self.right4 = UpConcatDoubleconv(1024, 512, xpad)
        self.right3 = UpConcatDoubleconv(512, 256, xpad)
        self.right2 = UpConcatDoubleconv(256, 128, xpad)
        self.right1 = UpConcatDoubleconv(128, 64, xpad)

        self.out = Outconv(64, Co)

    def forward(self, x):
        """
        Input:
            - x: UNet input; type: torch tensor; dimensions: [B, Ci, D, H, W]
        Output:
            - UNet output; type: torch tensor; dimensions: output[B, Co, D, H, W]
        Dimensions explained:
            - 'i' and 'o' subscripts repsectively mean inputs and outputs.
            - C: channels
            - D: depth
            - H: height
            - W: width
        """
        # Downsampling limb of UNet:
        x1 = self.left1(x)
        x2 = self.left2(x1)
        x3 = self.left3(x2)
        x4 = self.left4(x3)

        # Bottleneck of UNet:
        x = self.bottom(x4)

        # Upsampling limb of UNet:
        x = self.right4(x4, x)
        x = self.right3(x3, x)
        x = self.right2(x2, x)
        x = self.right1(x1, x)

        return self.out(x)


# -------------------------------------------------- UNet3D units ------------------------------------------------

class Doubleconv(nn.Module):
    """
    DoubleConvolution units in the 3D UNet
    """

    def __init__(self, Ci, Co):
        """
        Inputs:
            - Ci: number of input channels into the DoubleConvolution unit
            - Co: number of output channels out of the DoubleConvolution unit
        """
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv3d(Ci, Co, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(Co),
            nn.ReLU(inplace=True),
            nn.Conv3d(Co, Co, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(Co),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Input:
        - x: torch tensor; dimensions: [B, Ci, Di, Hi, Wi]
        Output:
             - return: x --> conv3d --> batch_norm --> ReLU --> conv3d --> batch_norm --> ReLU --> output
                dimensions: [B, Co, Do, Ho, Wo]
        """
        return self.doubleconv(x)

# ........................................................................................................

class DownDoubleconv(nn.Module):
    """
    Units in the left side of the 3D UNet:
    Down-sample using MaxPool3d --> then DoubleConvolution
    """

    def __init__(self, in_ch, out_ch):
        """
        Inputs:
            - Ci: number of input channels into the DownDoubleconv unit
            - Co: number of output channels out of the DownDoubleconv unit
        """
        super().__init__()
        self.maxpool_doubleconv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Doubleconv(in_ch, out_ch))


    def forward(self, x):
        """
        Input:
            - x: torch tensor; dimensions: x[batch, channels, D, H, W]
        Output:
            - return: x --> maxpool3d --> DoubleConv Unit --> output
        """
        return self.maxpool_doubleconv(x)

# ........................................................................................................

class UpConcatDoubleconv(nn.Module):
    """
    Units in the right side of the 3D UNet:
    Up-scale using ConvTranspose3d --> Concatenate the bottom and horizontal channels --> DoubleConvolution
    """

    def __init__(self, Ci, Co, xpad=True, up_mode='transposed'):
        """
        Inputs:
            - Ci: number of input channels into the UpConcatDoubleconv unit
            - Co: number of output channels out of the UpConcatDoubleconv unit
            - xpad: set this to False only if the input D/H/W dimensions are all powers of two. Otherwise set
                            this to True.
            - up_mode: default is 'transposed'. Set this to 'trilinear' if you want trilinear interpolation
                            instead (but interpolation would make the network slow).
        """
        super().__init__()
        self.xpad = xpad
        self.up_mode = up_mode

        if self.up_mode == 'transposed':
            self.up = nn.ConvTranspose3d(Ci, Co, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)

        self.doubleconv = Doubleconv(Ci, Co)


    def forward(self, x1, x2):
        """
        Inputs:
            - x1: skip-connection from the downsampling limb of U-Net; dimensions: [B, C1, D1, H1, W1]
            - x2: from the lower-level upsampling limb of U-Net; dimensions: [B, C2, D2, H2, W2]
        Output:
            - return: up-scale x2 --> concatenate(x1, x2) --> DoubleConv Unit --> output
        """
        x2 = self.up(x2)

        if self.xpad:
            # If H2 or W2 are smaller than H1 or W1, pad x2 so that its size matches x1.
            B1, C1, D1, H1, W1 = x1.shape
            B2, C2, D2, H2, W2 = x2.shape
            assert B1 == B2
            diffD, diffH, diffW = D1 - D2, H1 - H2, W1 - W2
            x2 = pad(x2, [diffW // 2, diffW - diffW // 2,
                          diffH // 2, diffH - diffH // 2,
                          diffD // 2, diffD - diffD // 2])

        # Concatenate x1 and x2:
        x = torch.cat([x1, x2], dim=1)

        # Return double convolution of the concatenated tensor:
        return self.doubleconv(x)

# ........................................................................................................

class Outconv(nn.Module):
    """
    Output unit in the 3D UNet
    """

    def __init__(self, Ci, Co):
        """
        Inputs:
            - in_ch: number of input channels into the final output unit
            - out_ch: number of output channels out of the entire UNet
        """
        super().__init__()
        self.conv_sigmoid = nn.Sequential(
            nn.Conv3d(Ci, Co, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv_sigmoid(x)



# --------------------------------------------- UNet3D class testing ---------------------------------------------
if __name__ == '__main__':

    from torchsummary import summary

    x = torch.rand(1, 1, 64, 64, 64)  # batch of 1 MRI volume: 1 channel, 64 x 64 x 64 voxels
    model = UNet3D()
    preds = model(x)
    print(f'Input shape: {x.shape} \n'
          f'Output shape: {preds.shape}')
    print(f'Input and output are the same shape? {preds.shape == x.shape}')

    summary(model, (1, 64, 64, 64))
    # for summary, the second argument is the shape of each input data (not the batch).
