import torch
import torch.nn as nn
import os
from networks_3D import MFN3D
from torch.nn import Linear, Conv3d, BatchNorm1d, BatchNorm3d, PReLU, Sequential, Module


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv3d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        #x = self.bn(x)
        x = self.conv(x)
        #x = self.bn(x)
        return x


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DDAMNet(nn.Module):
    def __init__(self, num_class=2, num_head=2, pretrained=True):
        super(DDAMNet, self).__init__()

        net = MFN3D.MixedFeatureNet3D()

        """if pretrained:
            net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))"""
        
        self.num_head = num_head
        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % i, CoordAttHead())

        self.Linear = Linear_block(512, 512, groups=512, kernel=(8, 8, 8), stride=(1, 1, 1), padding=(0, 0, 0))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

        self.features = nn.Sequential(*list(net.children())[:-4])

        

        
       
    def forward(self, x):
          
        x = self.features(x)
        
        heads = []

        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))
        head_out = heads  # Outputs delle attention heads

        y = heads[0]
        for i in range(1, self.num_head):
            y = torch.max(y, heads[i])

        y = x * y
        y = self.Linear(y)
        
        y = self.flatten(y)
        out = self.fc(y)
        return out, x, head_out


# Custom sigmoid
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # This ReLU clamps between 0 and 6

    def forward(self, x):
        return self.relu(x + 3) / 6


# Custom swish activation function
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt3D = CoordAtt3D(512, 512)

    def forward(self, x):
        ca = self.CoordAtt3D(x)
        return ca


class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt3D, self).__init__()

        self.Linear_t = Linear_block(inp, inp, groups=inp, kernel=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 8, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(1, 1, 8), stride=(1, 1, 1), padding=(0, 0, 0))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.conv2 = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(8, 8, 8), stride=(1, 1, 1), padding=(0, 0, 0))
        self.flatten = Flatten() 

    def forward(self, x):
        identity = x  
        # torch.Size([1, 512, 8, 8, 8]

        n, c, t, h, w = x.size()
        #1 8 8
        #8 1 8
        #8 8 1
        x_t = self.Linear_t(x).permute(0,1,3,2,4) #8,1,8
        x_h = self.Linear_h(x)                      #8,1,8
        x_w = self.Linear_w(x).permute(0,1,2,4,3)  #8,1,8                    

        # Concatenate features on the t dimension (dim=2)
        y = torch.cat([x_t, x_h, x_w], dim=2)
    
        # Apply convolutions and activations
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        split_size = [t, h, w]
        x_t, x_h, x_w = torch.split(y, split_size, dim=2)
        
        x_t = x_t.permute(0,1,3,2,4)
        x_h = x_h                     
        x_w = x_w.permute(0,1,2,4,3)

        
        x_t = self.conv4(x_t).sigmoid()
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        x_t = x_t.expand(-1, -1, t, h, w)
        x_h = x_h.expand(-1, -1, t, h, w)
        x_w = x_w.expand(-1, -1, t, h, w)
        
        # Combine the features
        out = identity * x_t * x_h * x_w

        return out
    