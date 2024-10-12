import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, in_chans, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, in_chans, out_chans, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_chans // 2  # hidden channels
        self.cv1 = Conv(in_chans, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_chans, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            spp_output = self.cv2(torch.cat((x,y1,y2,self.m(y2)),1))
            downsample_output = self.downsample(spp_output)
            return downsample_output

# input_data = torch.randn(1,3,64,64)
# model = SPPF(3,96)
# output = model(input_data)
# print(output.shape)
# net = SPPF(3,96)
# total_params = sum(p.numel() for p in net.parameters())
# flops, params = profile(net, inputs=(input_data,))
#
# print('the flops is {}G,the params is {}M'.format(round(flops/(10**9),2), round(params/(10**6),2)))