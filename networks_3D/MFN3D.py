from torch.nn import Linear, Conv3d, BatchNorm1d, BatchNorm3d, PReLU, Sequential, Module
import torch
import torch.nn as nn


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv3d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm3d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv3d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride,
                           padding=padding,
                           bias=False)
        self.bn = BatchNorm3d(out_c)

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.residual = residual

    def forward(self, x):
        
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt3D, self).__init__()
        self.pool_t = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.conv2 = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x  # Save original input
        n, c, t, h, w = x.size()

        # Apply adaptive pooling
        x_t = self.pool_t(x)
        x_h = self.pool_h(x).permute(0,1,3,2,4)                      
        x_w = self.pool_w(x).permute(0,1,4,3,2)

        # Concatenate pooled features on the height dimension (dim=2)
        y = torch.cat([x_t, x_h, x_w], dim=2)

        # Apply convolutions and activations
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        # Split features on the height dimension
        split_size = [t, h, w]
        x_t, x_h, x_w = torch.split(y, split_size, dim=2)
 
        x_h = x_h.permute(0,1,3,2,4)                      
        x_w = x_w.permute(0,1,4,3,2)

        # Apply final convolutions and expand to original size
        x_t = self.conv4(x_t).sigmoid()
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        # Permute back to original dimension order
        x_t = x_t.expand(-1, -1, t, h, w)
        x_h = x_h.expand(-1, -1, t, h, w)
        x_w = x_w.expand(-1, -1, t, h, w)
        
        # Combine the features
        out = identity * x_t * x_h * x_w

        return out


class MDConv3D(Module):
    def __init__(self, channels, kernel_size, split_out_channels, stride):
        super(MDConv3D, self).__init__()
        self.num_groups = len(kernel_size)
        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(Conv3d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))
        self.bn = BatchNorm3d(channels)
        self.prelu = PReLU(channels)

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)
        
        x_split = torch.split(x, self.split_channels, dim=1) 
        
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
             
        x = torch.cat(x, dim=1)
        
        return x


class Mix_Depth_Wise3D(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), groups=1,
                 kernel_size=[3, 5], split_out_channels=[64, 32, 32]):
        
        super(Mix_Depth_Wise3D, self).__init__()
        self.conv = Conv3d(in_c, groups, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.conv_dw = MDConv3D(channels=groups, kernel_size=kernel_size, split_out_channels=split_out_channels,
                                stride=stride)
        self.CA = CoordAtt3D(groups, groups)
        self.project = Conv3d(groups, out_c, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.residual = residual

    def forward(self, x):
        
        if self.residual:
            short_cut = x
        
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.CA(x)
        x = self.project(x)

        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual3D(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(Residual3D, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Mix_Depth_Wise3D(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups)
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Mix_Residual3D(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), kernel_size=[3, 5],
                 split_out_channels=[64, 64]):
        super(Mix_Residual3D, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Mix_Depth_Wise3D(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups,
                                 kernel_size=kernel_size, split_out_channels=split_out_channels))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MixedFeatureNet3D(Module):
    def __init__(self, embedding_size=256, out_t=8, out_h=8, out_w=8): #forse qua va messo 2 per le classi
        super(MixedFeatureNet3D, self).__init__()
        # Initial 3D Conv layers
        self.conv1 = Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv2_dw = Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=64)

        # Depth-wise and Mix Residual layers
        self.conv_23 = Mix_Depth_Wise3D(64, 64, kernel_size=[3, 5, 7], stride=(1, 2, 2), padding=(1, 1, 1), groups=128,
                                        split_out_channels=[64, 32, 32])
        self.conv_3 = Mix_Residual3D(64, num_block=9, groups=128, kernel_size=[3, 5], split_out_channels=[96, 32],
                                     kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv_34 = Mix_Depth_Wise3D(64, 128, kernel_size=[3, 5, 7], stride=(1, 2, 2), padding=(1, 1, 1), groups=256,
                                        split_out_channels=[128, 64, 64])
        self.conv_4 = Mix_Residual3D(128, num_block=16, groups=256, kernel_size=[3, 5], split_out_channels=[192, 64],
                                     kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv_45 = Mix_Depth_Wise3D(128, 256, kernel_size=[3, 5, 7, 9], stride=(1, 2, 2), padding=(1, 1, 1),
                                        groups=512*2, split_out_channels=[128 * 2, 128 * 2, 128 * 2, 128 * 2]) # groups=512 * 2 = 1024
        self.conv_5 = Mix_Residual3D(256, num_block=6, groups=512, kernel_size=[3, 5, 7],
                                     split_out_channels=[86 * 2, 85 * 2, 85 * 2],
                                     kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv_6_sep = Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv_6_dw = Conv3d(512, 512, groups=512, kernel_size=(out_t, out_h, out_w), stride=(1, 1, 1),
                                padding=(0, 0, 0))

        # Flatten and final Linear layers
        self.conv_6_flatten = nn.Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        print("starting MFN")
        out = self.conv1(x)
        
        out = self.conv2_dw(out)
        
        out = self.conv_23(out)
        
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)

        return l2_norm(out)