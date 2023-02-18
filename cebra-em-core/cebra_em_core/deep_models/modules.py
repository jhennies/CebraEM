
import torch.nn as nn
from torch import cat
from torch import optim
try:
    from torchsummary import summary
except:
    summary = None

ConvND = {2: nn.Conv2d, 3: nn.Conv3d}
ConvNDTranspose = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
MaxPoolingND = {2: nn.MaxPool2d, 3: nn.MaxPool3d}
BatchNormND = {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


class Downsampling(nn.Module):

    def __init__(
            self,
            num_convs,
            in_channels,  # The number of input channels for the first conv layer
            out_channels,  # int or List of output channels for each conv layer
            kernel_size,
            level,
            conv_strides=1,
            padding=1,
            activation='relu',
            batch_norm=None,
            ndims=2
    ):
        super(Downsampling, self).__init__()

        # Parameters
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.level = level
        self.conv_strides = conv_strides
        self.padding = padding
        self.activation = activation
        self.batch_norm = batch_norm
        self.ndims = ndims

        # _________________________
        # Layer initializations

        self.convolutions = nn.Sequential()
        in_ch = in_channels

        for idx in range(num_convs):
            if type(out_channels) is tuple:
                out_ch = out_channels[idx]
            else:
                out_ch = out_channels
            self.convolutions.add_module('conv{}'.format(idx), ConvND[ndims](
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=conv_strides,
                padding=padding
            ))

            if batch_norm:
                if type(batch_norm) is dict:
                    self.convolutions.add_module('bn{}'.format(idx), BatchNormND[ndims](out_ch, **batch_norm))
                else:
                    self.convolutions.add_module('bn{}'.format(idx), BatchNormND[ndims](out_ch))

            if activation == 'relu':
                self.convolutions.add_module('relu{}'.format(idx), nn.ReLU(inplace=True))
            else:
                raise NotImplementedError

            in_ch = out_ch

        self.max_pool = MaxPoolingND[ndims](2)

    def forward(self, x):

        x = self.convolutions(x)
        skip = x
        x = self.max_pool(x)
        return x, skip


class Upsampling(nn.Module):

    def __init__(
            self,
            num_convs,
            in_channels,  # The number of input channels for the first conv layer
            out_channels,  # int or List of output channels for each conv layer
            skip_channels,
            kernel_size,
            level,
            upsampling_size=2,
            upsampling_strides=2,
            conv_strides=1,
            padding=1,
            activation='relu',
            batch_norm=None,
            ndims=2
    ):
        super(Upsampling, self).__init__()

        # Parameters
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.level = level
        self.upsampling_size = upsampling_size
        self.upsampling_strides = upsampling_strides
        self.conv_strides = conv_strides
        self.padding = padding
        self.activation = activation
        self.batch_norm = batch_norm
        self.ndims = ndims

        # _________________________
        # Layer initializations

        self.conv_transpose = ConvNDTranspose[ndims](
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=upsampling_size,
            stride=upsampling_strides,
            padding=0
        )

        self.convolutions = nn.Sequential()

        in_ch = in_channels + skip_channels

        for idx in range(num_convs):
            if type(out_channels) is tuple:
                out_ch = out_channels[idx]
            else:
                out_ch = out_channels
            self.convolutions.add_module('conv{}'.format(idx), ConvND[ndims](
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=conv_strides,
                padding=padding
            ))

            if batch_norm:
                if type(batch_norm) is dict:
                    self.convolutions.add_module('bn{}'.format(idx), BatchNormND[ndims](out_ch, **batch_norm))
                else:
                    self.convolutions.add_module('bn{}'.format(idx), BatchNormND[ndims](out_ch))

            if activation == 'relu':
                self.convolutions.add_module('relu{}'.format(idx), nn.ReLU(inplace=True))
            else:
                raise NotImplementedError

            in_ch = out_ch

    def forward(self, x, skip):

        x = self.conv_transpose(x)
        x = cat((skip, x), dim=1)
        return self.convolutions(x)


class Bottleneck(nn.Module):

    def __init__(
            self,
            num_convs,
            in_channels,
            out_channels,
            kernel_size,
            conv_strides=1,
            padding=1,
            activation='relu',
            batch_norm=None,
            ndims=2
    ):
        super(Bottleneck, self).__init__()

        # Parameters
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_strides = conv_strides
        self.padding = padding
        self.activation = activation
        self.batch_norm = batch_norm
        self.ndims = ndims

        # _________________________
        # Layer initializations

        self.layers = nn.Sequential()
        in_ch = in_channels

        for idx in range(num_convs):
            if type(out_channels) is tuple:
                out_ch = out_channels[idx]
            else:
                out_ch = out_channels
            self.layers.add_module('conv{}'.format(idx), ConvND[ndims](
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=conv_strides,
                padding=padding
            ))

            if batch_norm:
                if type(batch_norm) is dict:
                    self.layers.add_module('bn{}'.format(idx), BatchNormND[ndims](out_ch, **batch_norm))
                else:
                    self.layers.add_module('bn{}'.format(idx), BatchNormND[ndims](out_ch))

            if activation == 'relu':
                self.layers.add_module('relu{}'.format(idx), nn.ReLU(inplace=True))
            else:
                raise NotImplementedError

            in_ch = out_ch

    def forward(self, x):
        return self.layers(x)


class Unet(nn.Module):

    def __init__(
            self,
            num_classes,
            in_channels,
            filter_sizes_down=((16, 32), (32, 64), (64, 128)),
            filter_sizes_up=((128, 128), (64, 64), (32, 32)),
            filter_sizes_bottleneck=(128, 256),
            kernel_size=3,
            batch_norm=None,
            ndims=2,
            return_last_upsampling=False,
            output_activation='sigmoid'
    ):
        super(Unet, self).__init__()

        # Parameters
        self.num_classes = num_classes
        self.filter_sizes_down = filter_sizes_down
        self.filter_sizes_up = filter_sizes_up
        self.filter_sizes_bottleneck = filter_sizes_bottleneck
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.ndims = ndims
        self.return_last_upsampling = return_last_upsampling
        self.output_activation = output_activation

        # _________________________________
        # Network layer initialization

        in_ch = in_channels
        self.downs = nn.ModuleList()
        for down_level in range(len(filter_sizes_down)):
            self.downs.append(Downsampling(
                num_convs=len(filter_sizes_down[down_level]),
                in_channels=in_ch,
                out_channels=filter_sizes_down[down_level],
                kernel_size=kernel_size,
                level=down_level,
                conv_strides=1,
                padding=1,
                activation='relu',
                batch_norm=batch_norm,
                ndims=ndims
            ))
            in_ch = filter_sizes_down[down_level][-1]

        self.bottleneck = Bottleneck(
            num_convs=len(filter_sizes_bottleneck),
            in_channels=in_ch,
            out_channels=filter_sizes_bottleneck,
            kernel_size=kernel_size,
            conv_strides=1,
            padding=1,
            activation='relu',
            batch_norm=batch_norm,
            ndims=ndims
        )
        in_ch = filter_sizes_bottleneck[-1]

        self.ups = nn.ModuleList()
        for up_level in range(len(filter_sizes_up)):
            self.ups.append(Upsampling(
                num_convs=len(filter_sizes_up[up_level]),
                in_channels=in_ch,
                out_channels=filter_sizes_up[up_level],
                skip_channels=filter_sizes_down[len(filter_sizes_up) - up_level - 1][-1],
                kernel_size=kernel_size,
                level=up_level,
                upsampling_strides=2,
                upsampling_size=2,
                conv_strides=1,
                padding=1,
                activation='relu',
                batch_norm=batch_norm,
                ndims=ndims
            ))
            in_ch = filter_sizes_up[up_level][-1]

        self.output_conv = ConvND[ndims](
            in_channels=in_ch,
            out_channels=num_classes,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        )
        if self.output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif self.output_activation == 'softmax':
            self.output_act = nn.Softmax()

    def forward(self, x):

        # Auto-encoder
        skips = []
        for down_level in range(len(self.filter_sizes_down)):

            x, skip = self.downs[down_level](x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Auto-decoder
        for up_level in range(len(self.filter_sizes_up)):

            x = self.ups[up_level](x, skips[len(self.filter_sizes_up) - up_level - 1])

        last_upsampling = x

        # Output layer
        x = self.output_conv(x)
        x = self.output_act(x)

        if not self.return_last_upsampling:
            return x
        else:
            return x, last_upsampling


if __name__ == '__main__':
    
    unet = Unet(1, 1, ndims=3)

    # optimizer = optim.Adam(unet.parameters(), 0.003)

    unet.cuda()
    if summary is not None:
        summary(unet, (1, 64, 64, 64))

