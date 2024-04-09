import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class Unet3D(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, feature_dim=None, nb_features=None):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [2, 3], "dims should be 2 or 3. found: %d" % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = ((8, 32, 32), (32, 32, 32, 8, 8, 3))

        self.enc_nf, self.dec_nf = nb_features
        extra_nf, prev_nf = 1, 1
        if feature_dim:
            extra_nf += feature_dim
            prev_nf += feature_dim

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")

        # configure encoder (down-sampling path)
        self.downarm = nn.ModuleList()
        for i, nf in enumerate(self.enc_nf):
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[: len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += extra_nf
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf) :]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.last_layer = Conv(self.dec_nf[-1], 1, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.last_layer.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.last_layer.weight.shape)
        )
        self.last_layer.bias = nn.Parameter(torch.zeros(self.last_layer.bias.shape))

    def forward(self, x, feat1=None, feat2=None):
        # get encoder activations
        if feat1 is None or feat2 is None:
            x_enc = [x]
        else:
            x_enc = [torch.cat([x, feat1, feat2], dim=1)]

        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        x = self.last_layer(x)

        return x


class Unet3D_multi(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, add_dim=(4, 8, 16)):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [2, 3], "dims should be 2 or 3. found: %d" % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = ((8, 16, 32), (32, 32, 32, 8, 8, 3))

        self.enc_nf, self.dec_nf = nb_features
        self.add_dim = add_dim
        extra_nf, prev_nf = 1, 1

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")

        # configure encoder (down-sampling path)
        self.downarm = nn.ModuleList()
        for i, nf in enumerate(self.enc_nf):
            if i == 0:
                self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            else:
                self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf + (self.add_dim[i] * 2)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        add_dim_history = list(reversed(self.add_dim))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[: len(self.enc_nf)]):
            channels = (
                prev_nf + enc_history[i] + (add_dim_history[i] * 2)
                if i > 0
                else prev_nf
            )
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += extra_nf
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf) :]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.last_layer = Conv(self.dec_nf[-1], 1, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.last_layer.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.last_layer.weight.shape)
        )
        self.last_layer.bias = nn.Parameter(torch.zeros(self.last_layer.bias.shape))

    def forward(self, x, feat_list_1, feat_list_2):
        # get encoder activations
        x_enc = [x]

        for idx, layer in enumerate(self.downarm):
            x_enc.append(
                torch.cat([layer(x_enc[-1]), feat_list_1[idx], feat_list_2[idx]], dim=1)
            )

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for idx, layer in enumerate(self.uparm):
            x = layer(x)
            if idx != len(self.uparm) - 1:
                x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        x = self.last_layer(x)

        return x


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """

        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv3d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv3d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.
        Parameters
        ----------
            x : tensor
                input to the NN block.
        Returns
        -------
            tensor
                output of the NN block.
        """

        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool3d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.
    ...
    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv3d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.
        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.
        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode="trilinear")
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class Unet3D_2(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(Unet3D_2, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv3d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv3d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv3d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.
        Parameters
        ----------
            x : tensor
                input to the UNet.
        Returns
        -------
            tensor
                output of the UNet.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x
