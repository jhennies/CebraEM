# from .modules import nn
import torch.nn as nn
from torch import cat
from cebra_em_core.deep_models.modules import Unet
try:
    from torchsummary import summary
except ImportError:
    summary = None


class cNnet(nn.Module):
    
    def __init__(
            self,
            n_nets=3,
            # default scale ratio: 2
            #
            #    EXAMPLE with three modules and the last with input shape = (64, 64, 64)
            #    First net input  (256, 256, 256) -> downsample -> (64, 64, 64)
            #    Second net input (128, 128, 128) -> downsample -> (64, 64, 64)
            #    Third net input  (64, 64, 64)
            #
            #    i.e. scale_ratio=2, n_nets=3: 256 -> 128 -> 64
            #         scale_ratio=4, n_nets=2: 256 -> 64
            scale_ratio=2,
            module_shape=(64, 64, 64),
            input_shapes=None,
            initial_ds=None,
            crop_and_ds_inputs=False,
            crop_and_us_outputs=True,
            in_channels=1,
            out_channels=None,
            num_inputs=1,
            filter_sizes_down=(
                    ((4, 8), (8, 16), (16, 32)),
                    ((8, 16), (16, 32), (32, 64)),
                    ((32, 64), (64, 128), (128, 256))
            ),
            filter_sizes_bottleneck=(
                    (32, 64),
                    (64, 128),
                    (256, 512)
            ),
            filter_sizes_up=(
                    ((32, 32), (16, 16), (8, 8)),
                    ((64, 64), (32, 32), (16, 16)),
                    ((256, 256), (128, 128), (64, 64))
            ),
            batch_norm=True,
            output_activation='softmax',
            verbose=False
    ):
        
        super(cNnet, self).__init__()

        # ______________________________
        # Parameters and settings

        # Assertions
        assert out_channels is not None, 'The number of output classes for each module needs to be specified!'
        assert in_channels is not None, 'The number of input channels needs to be specified!'
        assert len(filter_sizes_down) == n_nets
        assert len(filter_sizes_up) == n_nets
        assert len(filter_sizes_bottleneck) == n_nets

        # ______________________________
        # Define layers

        self.avg_pool_inputs = nn.ModuleDict()
        self.unets = nn.ModuleList()

        for net_idx in range(n_nets):

            if crop_and_ds_inputs:

                if 0 < net_idx < n_nets - 1:

                    # First, the tensor will be cropped
                    # and then downsampled
                    self.avg_pool_inputs[net_idx] = nn.AvgPool3d(
                        kernel_size=(scale_ratio ** (n_nets - net_idx - 1),) * 3
                    )

            self.unets.append(
                Unet(
                    num_classes=out_channels[net_idx],
                    in_channels=in_channels + out_channels[net_idx],
                    filter_sizes_down=filter_sizes_down[net_idx],
                    filter_sizes_up=filter_sizes_up[net_idx],
                    filter_sizes_bottleneck=filter_sizes_bottleneck,
                    kernel_size=3,
                    batch_norm=batch_norm,
                    ndims=3,
                    return_last_upsampling=True,
                    output_activation=output_activation
                )
            )

    def forward(self, x):
        
        # TODO: Define the architecture
        raise NotImplementedError

        output = x
        return output


class PiledUnet(nn.Module):
    """
    Special case of the cNnet where all nets have the same input scale.
    To achieve the same architecture using cNnet, set scale_ratio=1
    """

    def __init__(
            self,
            n_nets=3,
            in_channels=1,
            out_channels=None,
            filter_sizes_down=(
                    ((4, 8), (8, 16), (16, 32)),
                    ((8, 16), (16, 32), (32, 64)),
                    ((32, 64), (64, 128), (128, 256))
            ),
            filter_sizes_bottleneck=(
                    (32, 64),
                    (64, 128),
                    (256, 512)
            ),
            filter_sizes_up=(
                    ((32, 32), (16, 16), (8, 8)),
                    ((64, 64), (32, 32), (16, 16)),
                    ((256, 256), (128, 128), (64, 64))
            ),
            batch_norm=True,
            output_activation='softmax',
            predict=False
    ):

        super(PiledUnet, self).__init__()

        # ______________________________
        # Parameters and settings

        # Assertions
        assert out_channels is not None, 'The number of output classes for each module needs to be specified!'
        assert in_channels is not None, 'The number of input channels needs to be specified!'
        assert len(filter_sizes_down) == n_nets
        assert len(filter_sizes_up) == n_nets
        assert len(filter_sizes_bottleneck) == n_nets

        self.n_nets = n_nets
        self.predict = predict

        # ______________________________
        # Define layers

        self.unets = nn.ModuleList()

        for net_idx in range(n_nets):
            
            if net_idx == 0:
                in_ch = in_channels
            else:
                in_ch = in_channels + filter_sizes_up[net_idx][-1][-1]

            self.unets.append(
                Unet(
                    num_classes=out_channels[net_idx],
                    in_channels=in_ch,
                    filter_sizes_down=filter_sizes_down[net_idx],
                    filter_sizes_up=filter_sizes_up[net_idx],
                    filter_sizes_bottleneck=filter_sizes_bottleneck[net_idx],
                    kernel_size=3,
                    batch_norm=batch_norm,
                    ndims=3,
                    return_last_upsampling=True,
                    output_activation=output_activation
                )
            )

    def forward(self, in_x):

        # Define the architecture
        outputs = None
        out_x = None

        x = in_x

        for net_idx in range(self.n_nets):

            if net_idx > 0:
                x = cat((in_x, x), dim=1)

            out_x, x = self.unets[net_idx](x)

            if not self.predict:
                if net_idx == 0:
                    outputs = out_x
                else:
                    outputs = cat((outputs, out_x), dim=1)
        
        if self.predict:
            assert out_x is not None
            outputs = out_x

        return outputs
    
    
class MembraneNet(PiledUnet):
    """
    Just a piled unet with sensible parameters
    """

    def __init__(self, predict=False):

        super(MembraneNet, self).__init__(
            n_nets=3,
            in_channels=1,
            out_channels=[1, 1, 1],
            filter_sizes_down=(
                ((16, 32), (32, 64), (64, 128)),
                ((16, 32), (32, 64), (64, 128)),
                ((16, 32), (32, 64), (64, 128))
            ),
            filter_sizes_bottleneck=(
                (128, 256),
                (128, 256),
                (128, 256)
            ),
            filter_sizes_up=(
                ((256, 128, 128), (128, 64, 64), (64, 32, 32)),
                ((256, 128, 128), (128, 64, 64), (64, 32, 32)),
                ((256, 128, 128), (128, 64, 64), (64, 32, 32))
            ),
            batch_norm=True,
            output_activation='sigmoid',
            predict=predict
        )


class SlimMembraneNet(PiledUnet):
    """
    Just a piled unet with sensible parameters
    """

    def __init__(self, predict=False):

        super(SlimMembraneNet, self).__init__(
            n_nets=3,
            in_channels=1,
            out_channels=[1, 1, 1],
            filter_sizes_down=(
                ((8, 16), (16, 32), (32, 64)),
                ((8, 16), (16, 32), (32, 64)),
                ((8, 16), (16, 32), (32, 64))
            ),
            filter_sizes_bottleneck=(
                (64, 128),
                (64, 128),
                (64, 128)
            ),
            filter_sizes_up=(
                ((64, 64), (32, 32), (16, 16)),
                ((64, 64), (32, 32), (16, 16)),
                ((64, 64), (32, 32), (16, 16))
            ),
            batch_norm=True,
            output_activation='sigmoid',
            predict=predict
        )


if __name__ == '__main__':

    piled_unet = PiledUnet(
        n_nets=3,
        in_channels=1,
        out_channels=[1, 1, 1],
        filter_sizes_down=(
            ((16, 32), (32, 64), (64, 128)),
            ((16, 32), (32, 64), (64, 128)),
            ((16, 32), (32, 64), (64, 128))
        ),
        filter_sizes_bottleneck=(
            (128, 256),
            (128, 256),
            (128, 256)
        ),
        filter_sizes_up=(
            ((128, 128), (64, 64), (32, 32)),
            ((128, 128), (64, 64), (32, 32)),
            ((128, 128), (64, 64), (32, 32))
        ),
        batch_norm=True,
        output_activation='sigmoid'
    )

    # optimizer = optim.Adam(unet.parameters(), 0.003)

    piled_unet.cuda()
    if summary is not None:
        summary(piled_unet, (1, 64, 64, 64))

    pass
