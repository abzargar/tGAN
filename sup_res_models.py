from einops import rearrange
from torch import einsum
import torch.nn as nn
import functools
import torch
import torch.nn.functional as F

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)
class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         sequence += [Residual(PreNorm(ndf, LinearAttention(ndf)))]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)

import torch.nn as nn

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, dropout_rate=0.3):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True),
                    nn.Dropout(dropout_rate)]
        sequence += [Residual(PreNorm(ndf, LinearAttention(ndf)))]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(dropout_rate)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_rate)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

        # Store the sequence of layers in a ModuleList to extract intermediate features
        self.layers = nn.ModuleList(sequence) # Modify this line to have sequence as a ModuleList

    def forward(self, input, return_features=False):
        """Standard forward."""
        if return_features:
            # If return_features is True, collect and return intermediate features
            features = []
            x = input
            for layer in self.layers:
                x = layer(x)
                features.append(x)
            return features
        else:
            # Otherwise, just return the final output
            return self.model(input)





class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class en_conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class style_dc_conv_block(nn.Module):
    def __init__(self, in_c, out_c,latent_dim=128):
        super().__init__()

        self.to_style1 = nn.Linear(latent_dim, in_c)
        self.to_noise1 = nn.Linear(1, out_c)
        self.conv1 = Conv2DMod(in_c, out_c, 3)

        self.to_style2 = nn.Linear(latent_dim, out_c)
        self.to_noise2 = nn.Linear(1, out_c)
        self.conv2 = Conv2DMod(out_c, out_c, 3)
        self.activation = leaky_relu()

    def forward(self, x,istyle, inoise):
        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2))
        noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))
        style1 = self.to_style1(istyle)

        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = en_conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class style_decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = style_dc_conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip,istyle, inoise):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x,istyle, inoise)

        return x

class style_upsample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = style_dc_conv_block(out_c, out_c)

    def forward(self, inputs,istyle, inoise):
        x = self.up(inputs)
        x = self.conv(x,istyle, inoise)

        return x

class SupResStyleGenerator(nn.Module):
    def __init__(self,style_latent_dim=128,style_depth=3,style_lr_mul=0.1,output_nc=1):
        super().__init__()

        """ StyleNet """
        self.latent_dim=style_latent_dim
        self.StyleNet = StyleVectorizer(emb=style_latent_dim, depth=style_depth, lr_mul=style_lr_mul)

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = en_conv_block(512, 1024)

        """ Decoder """
        self.d1 = style_decoder_block(1024, 512)
        self.d2 = style_decoder_block(512, 256)
        self.d3 = style_decoder_block(256, 128)
        self.d4 = style_decoder_block(128, 64)
        self.d5 = style_upsample(64, 32)
        """ Classifier """
        self.outputs = nn.Conv2d(32, output_nc, kernel_size=1, padding=0)
        self.tanh=nn.Tanh()

    def latent_to_w(self,style_vectorizer, latent_descr):
        return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

    def styles_def_to_tensor(self,styles_def):
        return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

    def forward(self, inputs,style, input_noise):
        """ StyleNet """
        w_space = self.latent_to_w(self.StyleNet, style)
        w_styles = self.styles_def_to_tensor(w_space)

        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)
        styles = w_styles.transpose(0, 1)

        """ Decoder """
        d1 = self.d1(b, s4,styles[0], input_noise)
        d2 = self.d2(d1, s3,styles[1], input_noise)
        d3 = self.d3(d2, s2,styles[2], input_noise)
        d4 = self.d4(d3, s1,styles[3], input_noise)
        d5 = self.d5(d4, styles[4], input_noise)
        """ Classifier """
        outputs = self.outputs(d5)

        return self.tanh(outputs)

class DeepSeaUp(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1=self.double_conv(x)
        x2 = self.conv1d(x)
        x=x1+x2
        x=self.bn(x)
        x=self.relu(x)
        return x

class Segmentation(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Segmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res1=ResBlock(n_channels,64)
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256)
        self.up1 = DeepSeaUp(256, 128)
        self.res4 = ResBlock(384, 128)
        self.up2 = DeepSeaUp(128, 64)
        self.res5 = ResBlock(192, 64)
        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x1=self.res1(x)
        x2=self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)
        x4=self.up1(x3,x2)
        x4 = self.res4(x4)
        x5 = self.up2(x4,x1)
        x6 = self.res5(x5)
        logits=self.conv3(x6)
        return logits


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)



class StyleUnetSegmentation(nn.Module):
    def __init__(self,n_classes=2):
        super().__init__()
        self.n_classes=n_classes
        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = en_conv_block(512, 1024)
        self.feature_to_style = FeatureToStyle(1024, 5, 128)
        """ Decoder """
        self.d1 = style_decoder_segmentation_block(1024, 512)
        self.d2 = style_decoder_segmentation_block(512, 256)
        self.d3 = style_decoder_segmentation_block(256, 128)
        self.d4 = style_decoder_segmentation_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        # Convert bottleneck features to style features
        style_features = self.feature_to_style(b)

        # Decoder with bottleneck 'style' injection
        d1 = self.d1(b, s4, style_features[0])
        d2 = self.d2(d1, s3, style_features[1])
        d3 = self.d3(d2, s2, style_features[2])
        d4 = self.d4(d3, s1, style_features[3])

        # Classifier
        outputs = self.outputs(d4)

        return outputs




class style_decoder_segmentation_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = style_dc_conv_segmentation_block(out_c+out_c, out_c)

    def forward(self, inputs, skip,istyle):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x,istyle)

        return x


class style_dc_conv_segmentation_block(nn.Module):
    def __init__(self, in_c, out_c,latent_dim=128):
        super().__init__()

        self.to_style1 = nn.Linear(latent_dim, in_c)
        self.to_noise1 = nn.Linear(1, out_c)
        self.conv1 = Conv2DMod(in_c, out_c, 3)

        self.to_style2 = nn.Linear(latent_dim, out_c)
        self.to_noise2 = nn.Linear(1, out_c)
        self.conv2 = Conv2DMod(out_c, out_c, 3)
        self.activation = leaky_relu()

    def forward(self, x,istyle):
        style1 = self.to_style1(istyle)

        x = self.conv1(x, style1)
        x = self.activation(x)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x)
        return x


class FeatureToStyle(nn.Module):
    def __init__(self, in_channels, num_styles, latent_dim):
        super().__init__()
        self.num_styles = num_styles
        self.latent_dim = latent_dim
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, num_styles * latent_dim)

    def forward(self, x):
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten for FC
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape the output
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_styles, self.latent_dim)

        # Transpose to make it [num_styles, batch_size, latent_dim]
        x = x.permute(1, 0, 2)

        return x


class ComplexFlowNet(nn.Module):
    def __init__(self):
        super(ComplexFlowNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)  # 2 output channels for dx, dy flow

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)

    def forward(self, frame1, frame2):
        x = torch.cat([frame1, frame2], dim=1)  # Concatenate along channel dimension

        # Forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        flow = self.conv7(x)  # This will contain your flow in dx, dy

        return flow