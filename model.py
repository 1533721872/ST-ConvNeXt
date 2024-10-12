import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchstat import stat
#from conv_change import *
import torch.nn.utils.prune as prune
from SPPFPool import *
from SPConv import *

def drop_path(x, drop_prob: float = 0., training: bool = False):
    '''
    drop_path 是一种正则化手段，其效果是将深度学习模型中的多分支结构随机”删除“,以减轻过拟合问题。
    但在每个样本的路径上进行操作，而不是在特定层的输出上操作。它在训练期间随机丢弃一些路径，
    类似于随机深度的概念。这可以被看作是在训练期间减少网络的深度，从而提供一种正则化效果。
    '''
    # x表示张量，drop_prob丢弃的概率。
    # training: bool = False，
    # 如果为False，或者drop_prob为0，则不执行"Drop Path"操作，直接返回输入张量。
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob # 保留路径的概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 创建一个与输入张量x具有相同形状的随机张量random_tensor，其中元素的值是从0到1之间的随机数。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 调用floor函数，即将小于1-keep_prob的值设置为0，大于等于1-keep_prob的值设置为1，实现随机丢弃路径的效果。
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    '''
    将之前提到的drop_path函数封装成PyTorch的nn.Module的子类DropPath
    '''
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def shuffle_channels(x, groups):
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

# 尺寸减少4倍
class ConvTokenizer(nn.Module):
    def __init__(self, embedding_dim):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, embedding_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, embedding_dim // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1))
        )

    def forward(self, x):
        return self.block(x)


class SE_Block(nn.Module):  # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# 定义一个COnvNext中的block块，以用于构建ConvNext网络
class Block(nn.Module):
    def __init__(self, dim, drop_rate=0.,layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = SPConv_3x3(dim,dim,stride=1,ratio=0.5)
        # self.norm = nn.BatchNorm2d(num_features=dim)
        self.norm = LayerNorm(dim,eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Conv2d(dim,dim,kernel_size=7,stride=1,padding=3,groups=dim)
        self.pwconv2 = nn.Conv2d(dim,4*dim,kernel_size=3,stride=1,padding=1,groups=dim)
        self.act = nn.GELU()
        #self.act = nn.ReLU()
        #self.act = nn.Hardswish()
        self.pwconv3 = nn.Conv2d(4*dim, dim, kernel_size=3, stride=1,padding=1,groups=dim)
        self.se = SE_Block(dim)
        #self.pwconv4 = nn.Conv2d(2*dim, dim,kernel_size=7,stride=1,padding=3,groups=dim) # 维度还原
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity() # 正则化

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 输入一个x，返回一个张量
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        x = self.pwconv3(x)
        #x = self.pwconv4(x)
        # [N, H, W, C] -> [N, C, H, W]
        x = shuffle_channels(x, groups=2)
        x = shortcut + self.drop_path(x)
        x = self.act(x)
        return x


class ConvNeXt(nn.Module):

    def __init__(self, in_chans: int = 3, num_classes: int = 21, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0.,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(ConvTokenizer(dims[0]),
                             #nn.Conv2d(in_chans,dims[0],kernel_size=4,stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        # stem = nn.Sequential(GhostModuleV2(in_chans,dims[0],kernel_size=4,stride=4, ratio=2, dw_size=3,mode='attn'),
        #                      nn.BatchNorm2d(num_features=dims[0],eps=1e-6))
        # stem = nn.Sequential(SPPF(dims[0],dims[0]))
        self.downsample_layers.append(stem)
        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             SPPF(dims[i], dims[i+1]),
                                             # nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                                             )
            # downsample_layer = nn.Sequential(SPPF(dims[i],dims[i]))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.BatchNorm2d(dims[-1])  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.sppf = SPPF(dims[-1],dims[-1])


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            #print(f"Layer {i + 1} output shape: {x.shape}")
            x = self.stages[i](x)
            #print(f"Layer {i + 1} output shape: {x.shape}")

        return self.norm(x)  #
        # return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.mean([-2,-1]) #全局平均池化
        x = self.head(x)
        #print(f"Model output shape: {x.shape}")
        return x


def convnext_tiny(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


# input_data = torch.randn(1,3,224,224)
# model = ConvNeXt(in_chans=3, num_classes=21,
#                  depths=[3, 3, 9, 3],
#                  dims=[96, 192, 384, 768])
# output = model(input_data)
# print(output.shape)
# tensorboard.exe --logdir=C:/Users/administered/Desktop/ST-ConvNeXt/log

