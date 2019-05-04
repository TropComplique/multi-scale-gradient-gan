import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class EqualLR:

    def __init__(self, name, scaler):
        self.name = name
        self.scaler = scaler

    def scale(self, module):
        weight = getattr(module, self.name + '_unscaled')
        return weight * self.scaler

    @staticmethod
    def apply(module, name):

        weight = getattr(module, name)
        del module._parameters[name]

        weight = nn.Parameter(weight.data)
        module.register_parameter(name + '_unscaled', weight)
        shape = weight.data.shape

        if isinstance(module, nn.Conv2d):
            fan_in = shape[1] * shape[2] * shape[3]
        elif isinstance(module, nn.Linear):
            fan_in = shape[1]
        elif isinstance(module, nn.ConvTranspose2d):
            fan_in = shape[0]

        scaler = math.sqrt(2.0 / fan_in)
        hook = EqualLR(name, scaler)
        module.register_forward_pre_hook(hook)

    def __call__(self, module, input):
        weight = self.scale(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class Conv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        layer = nn.Conv2d(*args, **kwargs)
        init.normal_(layer.weight)
        init.zeros_(layer.bias)
        self.layer = equal_lr(layer)

    def forward(self, x):
        return self.layer(x)


class Linear(nn.Module):

    def __init__(self, in_dimension, out_dimension):
        super().__init__()

        layer = nn.Linear(in_dimension, out_dimension)
        init.normal_(layer.weight)
        init.zeros_(layer.bias)
        self.layer = equal_lr(layer)

    def forward(self, x):
        return self.layer(x)


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, c, h, w].
        Returns:
            a float tensor with shape [b, c, h, w].
        """
        s = torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return x * s


class MinibatchStdDev(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, c, h, w].
        Returns:
            a float tensor with shape [b, c + 1, h, w].
        """
        b, _, h, w = x.shape

        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0))
        # it has shape [c, h, w]

        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(b, 1, h, w)
        x = torch.cat([x, y], dim=1)
        return x


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_channels, z_dimension):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.linear = Linear(z_dimension, 2 * in_channels)

        self.linear.layer.bias.data[:in_channels] = 1.0
        self.linear.layer.bias.data[in_channels:] = 0.0

    def forward(self, x, z):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
            w: a float tensor with shape [b, z_dimension].
        Returns:
            a float tensor with shape [b, in_channels, h, w].
        """

        s = self.linear(z).unsqueeze(2).unsqueeze(3)
        gamma, beta = s.chunk(2, dim=1)
        # they have shape [b, in_channels, 1, 1]

        x = self.norm(x)
        return gamma * x + beta


class NoiseInjection(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        scaler = torch.zeros(1, in_channels, 1, 1)
        self.scaler = nn.Parameter(scaler)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, in_channels, h, w].
        """
        b, _, h, w = x.shape
        noise = torch.randn(b, 1, h, w, device=x.device)
        return x + self.scaler * noise


class InitialGeneratorBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        transposed = nn.ConvTranspose2d(num_channels, num_channels, 4)
        init.normal_(transposed.weight)
        init.zeros_(transposed.bias)

        self.layers = nn.Sequential(
            equal_lr(transposed),
            nn.LeakyReLU(0.2),
            Conv2d(num_channels, num_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, num_channels].
        Returns:
            a float tensor with shape [b, num_channels, 4, 4].
        """

        x = z.unsqueeze(2).unsqueeze(3)
        # it has shape [b, num_channels, 1, 1]

        return self.layers(x)


class GeneratorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, z_dimension=512):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise1 = NoiseInjection(out_channels)
        self.relu1 = nn.LeakyReLU(0.2)
        self.adain1 = AdaptiveInstanceNorm(out_channels, z_dimension)

        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.noise2 = NoiseInjection(out_channels)
        self.relu2 = nn.LeakyReLU(0.2)
        self.adain2 = AdaptiveInstanceNorm(out_channels, z_dimension)

    def forward(self, x, z):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
            z: a float tensor with shape [b, z_dimension].
        Returns:
            a float tensor with shape [b, out_channels, 2 * h, 2 * w].
        """
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv1(x)
        x = self.noise1(x)
        x = self.relu1(x)
        x = self.adain1(x, z)

        x = self.conv2(x)
        x = self.noise2(x)
        x = self.relu2(x)
        x = self.adain2(x, z)

        return x


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.layers(x)


class FinalDiscriminatorBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.layers = nn.Sequential(
            MinibatchStdDev(),
            Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            Conv2d(in_channels, in_channels, 4),
            nn.LeakyReLU(0.2),
            Conv2d(in_channels, 1, 1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, 4, 4].
        Returns:
            a float tensor with shape [b].
        """
        b = x.shape[0]
        return self.layers(x).view(b)


class Generator(nn.Module):

    def __init__(self, depth=6):
        super().__init__()

        assert depth >= 5
        z_dimension = 512

        progression = [InitialGeneratorBlock(z_dimension)]
        to_rgb = [Conv2d(z_dimension, 3, 1)]

        for i in range(depth):

            in_channels = min(2 ** (4 + depth - i), 512)
            out_channels = min(2 ** (4 + depth - i - 1), 512)

            block = GeneratorBlock(in_channels, out_channels)
            converter = Conv2d(out_channels, 3, 1)

            progression.append(block)
            to_rgb.append(converter)

        """
        If depth = 6 then tuples
        (i, output shape of a block) are:
        0, [b, 512, 8, 8]
        1, [b, 256, 16, 16]
        2, [b, 128, 32, 32]
        3, [b, 64, 64, 64]
        4, [b, 32, 128, 128]
        5, [b, 16, 256, 256]
        """

        self.progression = nn.ModuleList(progression)
        self.to_rgb = nn.ModuleList(to_rgb)
        self.depth = depth

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, z_dimension].
        Returns:
            a list with float tensors. Where `i`-th tensor
            has shape [b, 3, s, s] with s = 4 * (2 ** i).
            Integer `i` is in range [0, depth].
        """

        x = self.progression[0](z)
        outputs = [self.to_rgb[0](x)]

        for i in range(self.depth):

            x = self.progression[i + 1](x, z)
            outputs.append(self.to_rgb[i + 1](x))

        return outputs


class Discriminator(nn.Module):

    def __init__(self, depth=6, max_channels=512):
        super().__init__()

        assert depth >= 5
        from_rgb = [Conv2d(3, 16, 1)]
        progression = [DiscriminatorBlock(16, 32)]

        for i in range(1, depth):

            in_channels = min(2 ** (4 + i), max_channels)
            out_channels = min(2 ** (4 + i + 1), max_channels)

            from_rgb.append(Conv2d(3, 16, 1))
            progression.append(DiscriminatorBlock(in_channels + 16, out_channels))

        """
        If depth = 6 and max_channels = 512 then
        tuples (i, output shape of a block) are:
        1, [b, 64, 64, 64]
        2, [b, 128, 32, 32]
        3, [b, 256, 16, 16]
        4, [b, 512, 8, 8]
        5, [b, 512, 4, 4]
        """

        self.final_from_rgb = Conv2d(3, 16, 1)
        self.final_block = FinalDiscriminatorBlock(out_channels + 16)

        self.progression = nn.ModuleList(progression)
        self.from_rgb = nn.ModuleList(from_rgb)
        self.depth = depth

    def forward(self, inputs):
        """
        Arguments:
            inputs: a list with float tensors. Where `i`-th tensor
            has shape [b, 3, s, s] with s = 4 * (2 ** i).
            Integer `i` is in range [0, depth].
        Returns:
            a float tensor with shape [b].
        """
        depth = self.depth

        x = inputs[depth]
        x = self.from_rgb[0](x)
        # it has shape [b, 16, s, s],
        # where s = 4 * (2 ** depth)

        x = self.progression[0](x)
        # it has shape [b, 32, s / 2, s / 2]

        for i in range(1, depth):

            f = self.from_rgb[i](inputs[depth - i])
            x = torch.cat([x, f], dim=1)

            # x has spatial size s x s,
            # where s = 4 * 2 ** (depth - i).

            x = self.progression[i](x)

        f = self.final_from_rgb(inputs[0])
        x = torch.cat([x, f], dim=1)
        # it has spatial size 4 x 4

        return self.final_block(x)
