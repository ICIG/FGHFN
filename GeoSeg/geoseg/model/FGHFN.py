import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import math
from einops.layers.torch import Rearrange
#from pytorch_wavelets import DWTForward
import numbers
from einops import rearrange


BN_MOMENTUM = 0.1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0.1,
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}



def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model



def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model



def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], mode="base", **kwargs, )
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class MakeLayerNorm(nn.Module):
    def __init__(self, dim):
        super(MakeLayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Mlp(nn.Module):
    def __init__(self, channels, mlp_rate=4, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        hidden_channels = channels * mlp_rate
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim//2, hidden_dim//2, groups=hidden_dim//2, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.conv1x1_2 = nn.Conv2d(hidden_dim//2, dim, 1, 1, 0)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1x1(x)
        x_1, x_2 = x.chunk(2, dim=1)
        x_1 = self.dwconv(x_1)
        x = x_1 * x_2
        x = self.conv1x1_2(x)
        return x

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)




class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim//2, hidden_dim//2, groups=hidden_dim//2, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.conv1x1_2 = nn.Conv2d(hidden_dim//2, dim, 1, 1, 0)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.partial_conv3 = nn.Conv2d(dim//4, dim//4, 3, 1, 1, bias=False)

    def forward(self, x):
        x1,x2,x3,x4 = x.chunk(4, dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat([x1, x2,x3,x4],dim=1)
        x = self.conv1x1(x)
        x_1, x_2 = x.chunk(2, dim=1)
        x_1 = self.dwconv(x_1)
        x = x_1 * x_2
        x = self.conv1x1_2(x)
        return x
    
    
class HRFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.down128_64 = nn.Sequential(
            nn.BatchNorm2d(channels[0]),
            nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2)
        )
        self.down64_32 = nn.Sequential(
            nn.BatchNorm2d(channels[1]),
            nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2)
        )
        self.up64_128 = UpSample(channels[1], channels[0], scale_factor=2)

        self.up32_64 = UpSample(channels[2], channels[1], scale_factor=2)

        self.up16_32 = UpSample(channels[3], channels[2], scale_factor=2)


        self.BasicBlock128 = nn.Sequential(
            BasicBlock(channels[0]),
            BasicBlock(channels[0])
        ) 
        self.BasicBlock64 = nn.Sequential(
            BasicBlock(channels[1]),
            BasicBlock(channels[1])
        ) 
        self.BasicBlock32 = nn.Sequential(
            BasicBlock(channels[2]),
            BasicBlock(channels[2])
        ) 
        self.psi128 = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.psi64 = nn.Sequential(
            nn.Conv2d(channels[1], 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.psi32 = nn.Sequential(
            nn.Conv2d(channels[2], 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x128, x64, x32, x16):
        x128_64 = self.down128_64(x128)

        x64_32 = self.down64_32(x64)
        x64_128 = self.up64_128(x64)

        x32_64 = self.up32_64(x32)

        x16_32 = self.up16_32(x16)

        out128 =self.BasicBlock128(x128 + x64_128)
        out64 = self.BasicBlock64(x64 + x128_64 + x32_64)
        out32 = self.BasicBlock32(x32 + x64_32 + x16_32)

        out128 = x128 * self.psi128(out128)
        out64 = x64 * self.psi64(out64)
        out32 = x32 * self.psi32(out32)

        return out128, out64, out32


class LocalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_init = nn.Sequential(  # PW
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3//2, groups=channels),
            nn.BatchNorm2d(channels)
        )
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(channels//2, channels//2, kernel_size=5, stride=1,
                       padding=5//2, dilation=1, groups=channels//2),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # PW
            nn.Conv2d(channels//2, channels//2, kernel_size=7, stride=1,
                       padding=7//2, dilation=1, groups=channels//2),
            nn.GELU()
        )
        self.dwc = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3//2, groups=channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3//2, groups=channels),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        shortcut = self.dwc(x)
        x = self.conv_init(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv_init_1(x1)
        x2 = self.conv_init_2(x2)
        x = self.conv_out(torch.cat([shortcut, x1, x2], dim=1))
        return x


class GlobalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.patch_size = 8
        self.trans = nn.Conv2d(channels, channels*2, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels*2)
        self.conv_layer = nn.Conv2d(channels*2, channels*2, kernel_size=1, groups=channels*2)
        # self.psi = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.trans(x)
        q, k = x.chunk(2, dim=1)
        batch, c, h, w = x.size()

        # q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q,  norm='ortho')
        k_fft = torch.fft.rfft2(k, norm='ortho')

        out = q_fft * k_fft

        x_fft_real = torch.unsqueeze(torch.real(out), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(out), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])


        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.norm(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        out = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return out


class LayerNorm2(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Fuse(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.proj_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
        )
        self.strip1 = nn.Sequential(
            nn.Conv2d(channels//2, channels//2, (1, 7), padding=(0, 3), groups=channels//2),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels//2, (7, 1), padding=(3, 0), groups=channels//2),
        )
        self.strip2 = nn.Sequential(
            nn.Conv2d(channels//2, channels//2, (1, 5), padding=(0, 5//2), groups=channels//2),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels//2, (5, 1), padding=(5//2, 0), groups=channels//2),

         )

        self.proj_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x, y):
        attn = self.proj_1(x + y)
        x1, x2 = attn.chunk(2, dim=1)
        attn1 = self.strip1(x1)
        attn2 = self.strip2(x2)
        attn = self.proj_2(torch.cat([attn1 , attn2], dim=1))
        out = attn*0.2 + x + y
        out = self.proj_out(out)
        return out


    
class DecoderBlock(nn.Module):
    def __init__(self, channels, mlp_rate=2, drop=0.):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm2(channels, eps=1e-6, data_format="channels_first")
        self.local_block = LocalBlock(channels)
        self.global_block = GlobalBlock(channels)
        self.mlp = Mlp(channels, mlp_rate=mlp_rate)

    def forward(self, x):
        shortcut = x

        l = self.local_block(x)

        g = self.global_block(x)
        attn = l * g
        attn = self.norm(attn)
        out = self.mlp(attn) + shortcut

        return out



class Model(nn.Module):
    def __init__(self, n_class=6, pretrained=True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        config = [96, 192, 384, 768]  # channles of convnext-small
        self.backbone = convnext_small(pretrained, True)

        self.seg = nn.Sequential(
            nn.Conv2d(config[0], config[0] // 2, 1),
            nn.Conv2d(config[0] // 2, config[0] // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(config[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config[0] // 2, n_class, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

        self.up64_128 = UpSample(config[1], config[0], scale_factor=2)
        self.up16_32 = UpSample(config[3], config[2], scale_factor=2)
        self.up32_64 = UpSample(config[2], config[1], scale_factor=2)

        self.decoder_block16 = DecoderBlock(config[3], mlp_rate=2)
        self.decoder_block32 = DecoderBlock(config[2], mlp_rate=2)
        self.decoder_block64 = DecoderBlock(config[1], mlp_rate=2)
        self.decoder_block128 = DecoderBlock(config[0], mlp_rate=2)


        self.fuse32 = Fuse(config[2])
        self.fuse64 = Fuse(config[1])
        self.fuse128 = Fuse(config[0])

        self.HRFusion = HRFusionModule(config)

    def forward(self, x):
        x128, x64, x32, x16 = self.backbone(x)

        mid128, mid64, mid32= self.HRFusion(x128, x64, x32, x16)

        out16 = self.decoder_block16(x16)
        out16_32 = self.up16_32(out16)

        out32 = self.fuse32(mid32, out16_32)
        out32 = self.decoder_block32(out32)
        out32_64 = self.up32_64(out32)

        out64 = self.fuse64(mid64, out32_64)
        out64 = self.decoder_block64(out64)
        out64_128 = self.up64_128(out64)
        
        out128 = self.fuse128(mid128, out64_128)
        out128 = self.decoder_block128(out128)

        out = self.seg(out128)
        return out


