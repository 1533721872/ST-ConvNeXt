import torch
import torch.nn as nn
from thop import profile
class SPConv_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, ratio=0.5):
        super(SPConv_3x3, self).__init__()
        '''
        3x3卷积分支的输入通道数。
        它的值是inplanes（构造函数中传入的输入通道数）与ratio相乘后取整数部分。
        这表示3x3卷积分支将使用输入通道的一部分。
        1x1卷积分支的输入通道数。
        它的值是inplanes减去3x3卷积分支的输入通道数，以确保总输入通道数不变。
        '''
        self.inplanes_3x3 = int(inplanes*ratio)
        self.inplanes_1x1 = inplanes - self.inplanes_3x3
        self.outplanes_3x3 = int(outplanes*ratio)
        self.outplanes_1x1 = outplanes - self.outplanes_3x3
        self.outplanes = outplanes
        self.stride = stride
        '''
        执行k*k卷积提取重要特征
        '''
        self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
                             padding=1, groups=2, bias=False)
        self.pwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=1, bias=False)
        '''
        执行1x1卷积层补充隐含细节信息
        '''
        self.conv1x1 = nn.Conv2d(self.inplanes_1x1, self.outplanes,kernel_size=1,groups=self.inplanes_1x1)

        self.avgpool_s2_1 = nn.AvgPool2d(kernel_size=2,stride=2) #尺寸减半
        self.avgpool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_add_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)
        self.ratio = ratio
        #self.groups = int(1/self.ratio)

    def forward(self, x):
        b, c, _, _ = x.size() #获取输入x的尺寸信息
        # print(x.shape)
        x_3x3 = x[:,:int(c*self.ratio),:,:]
        # print(x_3x3.shape)#输入x变成两部分：x_3x3，x_1x1
        x_1x1 = x[:,int(c*self.ratio):,:,:]
        # print(x_1x1.shape)
        out_3x3_gwc = self.gwc(x_3x3)
        # print(out_3x3_gwc.shape)
        if self.stride ==2:
            x_3x3 = self.avgpool_s2_3(x_3x3) # 如果s=2，进行2倍下采样，即平均池化层
        out_3x3_pwc = self.pwc(x_3x3)
        out_3x3 = out_3x3_gwc + out_3x3_pwc
        out_3x3 = self.bn1(out_3x3)
        out_3x3_ratio = self.avgpool_add_3(out_3x3).squeeze(dim=3).squeeze(dim=2)

        # use avgpool first to reduce information lost
        if self.stride == 2:
            x_1x1 = self.avgpool_s2_1(x_1x1)

        out_1x1 = self.conv1x1(x_1x1)
        out_1x1 = self.bn2(out_1x1)
        out_1x1_ratio = self.avgpool_add_1(out_1x1).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
        '''
        使用 Softmax 权重将1x1卷积分支和3x3卷积分支的输出线性组合，得到最终的输出 out。
        权重是根据 out_31_ratio 计算的，分别用于两个分支的输出。
        '''
        out = out_1x1 * (out_31_ratio[:,:,1].view(b, self.outplanes, 1, 1).expand_as(out_1x1))\
              + out_3x3 * (out_31_ratio[:,:,0].view(b, self.outplanes, 1, 1).expand_as(out_3x3))

        return out

# input_data = torch.randn(1,96,224,224)
# net = SPConv_3x3(inplanes=96,outplanes=96,stride=1,ratio=0.5)
# output = net(input_data)
# print(output.shape)
# # net = nn.Conv2d(64,128,kernel_size=3,stride=1)
# total_params = sum(p.numel() for p in net.parameters())
# flops, params = profile(net, inputs=(input_data,))
#
# print('the flops is {}G,the params is {}M'.format(round(flops/(10**9),2), round(params/(10**6),2)))