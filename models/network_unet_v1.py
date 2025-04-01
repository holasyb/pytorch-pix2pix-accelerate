import torch

import torch.nn as nn
from torch.nn import init


class UnetEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        norm="batch",
        use_act=True,
        use_norm=True,
        use_bias=False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=use_bias
        )
        self.use_norm = use_norm
        self.use_act = use_act
        if use_norm:
            if norm == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
                # self.norm = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
            elif norm == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None
        if use_act:
            # self.activation = nn.ReLU()
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UnetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        norm="batch",
        use_act=True,
        use_norm=True,
        use_bias=False,
    ):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=use_bias
        )
        if use_norm:
            if norm == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None

        if use_act:
            # self.activation = nn.ReLU()
            self.activation = nn.ReLU(inplace=True)

        self.use_norm = use_norm
        self.use_act = use_act

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.activation(x)
        return x


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(
        self,
        block_expansion=64,
        in_features=3,
        num_blocks=6,
        max_features=512,
    ):
        super(Encoder, self).__init__()

        self.downsample_blocks = nn.ModuleList()
        temp_layer = UnetEncoderBlock(
            in_features,
            min(max_features, block_expansion),
            kernel_size=4,
            padding=1,
            norm="batch",
            use_norm=False,
            use_act=False,
            use_bias=False,
        )
        self.downsample_blocks.append(temp_layer)

        use_bias = False
        for i in range(1, num_blocks):
            if i == num_blocks - 1:
                use_norm = True
                use_act = True
            else:
                use_norm = True
                use_act = True
            temp_input_ch = min(max_features, block_expansion * (2 ** (i - 1)))
            temp_output_ch = min(max_features, block_expansion * (2**i))
            temp_layer = UnetEncoderBlock(
                temp_input_ch,
                temp_output_ch,
                kernel_size=4,
                padding=1,
                norm="batch",
                use_act=use_act,
                use_norm=use_norm,
                use_bias=use_bias,
            )
            self.downsample_blocks.append(temp_layer)

    def forward(self, x):
        outs = [x]
        for block in self.downsample_blocks:
            outs.append(block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(
        self,
        block_expansion=64,
        in_features=512,
        out_features=3,
        num_blocks=3,
        max_features=512,
    ):
        super(Decoder, self).__init__()

        self.upsample_blocks = nn.ModuleList()
        temp_input_ch = min(max_features, block_expansion * (2 ** (num_blocks - 1)))
        temp_output_ch = min(max_features, block_expansion * (2 ** (num_blocks - 2)))
        temp_layer = UnetDecoderBlock(
            temp_input_ch,
            temp_output_ch,
            kernel_size=4,
            padding=1,
            norm="batch",
            use_act=True,
            use_norm=True,
            use_bias=False,
        )
        self.upsample_blocks.append(temp_layer)  # 512x2hx2w

        for i in range(1, num_blocks - 1)[::-1]:
            in_filters = 2 * min(max_features, block_expansion * (2**i))
            out_filters = min(max_features, block_expansion * (2 ** (i - 1)))

            temp_layer = UnetDecoderBlock(
                in_filters,
                out_filters,
                kernel_size=4,
                padding=1,
                norm="batch",
                use_act=True,
                use_norm=True,
                use_bias=False,
            )
            self.upsample_blocks.append(temp_layer)

        temp_layer = UnetDecoderBlock(
            out_filters * 2,
            out_features,
            kernel_size=4,
            padding=1,
            norm="batch",
            use_act=False,
            use_norm=False,
            use_bias=True,
        )
        self.upsample_blocks.append(temp_layer)  # 512x2hx2w

    def forward(self, x):
        out = x.pop()
        for up_block in self.upsample_blocks:
            out = up_block(out)
            if not x:
                break
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Unet_v1(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_expansion=64,
        max_ch=512,
        norm="batch",
        layer_num=6,
    ):
        super(Unet_v1, self).__init__()
        self.encoder = Encoder(
            block_expansion=block_expansion,
            in_features=in_channels,
            num_blocks=layer_num,
            max_features=max_ch,
        )
        self.decoder = Decoder(
            block_expansion=block_expansion,
            in_features=max_ch,
            num_blocks=layer_num,
            max_features=max_ch,
        )

    def forward(self, x):
        enc_outs = self.encoder(x)
        dec_out = self.decoder(enc_outs[1:])
        return dec_out


if __name__ == "__main__":
    model = Unet_v1(
        in_channels=3,
        out_channels=3,
        block_expansion=64,
        max_ch=512,
        norm="batch",
        layer_num=8,
    )
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(output.shape)  # Should be [1, 3, 256, 256]
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {params}")
