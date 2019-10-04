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


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_channels, z_dimension):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.layers = nn.Sequential(
            Linear(z_dimension, 2 * in_channels),
            nn.ReLU(inplace=True)
            Linear(2 * in_channels, 2 * in_channels)
        )

    def forward(self, x, z):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
            z: a float tensor with shape [b, z_dimension].
        Returns:
            a float tensor with shape [b, in_channels, h, w].
        """

        w = self.layers(z).unsqueeze(2).unsqueeze(3)
        gamma, beta = w.chunk(2, dim=1)
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

        x += self.scaler * noise
        return x


class InitialGeneratorBlock(nn.Module):

    def __init__(self, initial_size, out_channels, z_dimension):
        super().__init__()

        h, w = initial_size
        constant = torch.randn(1, out_channels, h, w)
        self.constant = nn.Parameter(constant)

        self.noise1 = NoiseInjection(out_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.adain1 = AdaptiveInstanceNorm(out_channels, z_dimension)

        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.noise2 = NoiseInjection(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.adain2 = AdaptiveInstanceNorm(out_channels, z_dimension)

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, z_dimension].
        Returns:
            a float tensor with shape [b, out_channels, h, w].
        """
        b = z.size(0)
        x = self.constant.repeat(b, 1, 1, 1)

        x = self.noise1(x)
        x = self.relu1(x)
        x = self.adain1(x, z)

        x = self.conv2(x)
        x = self.noise2(x)
        x = self.relu2(x)
        x = self.adain2(x, z)

        return x


class GeneratorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, z_dimension):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise1 = NoiseInjection(out_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.adain1 = AdaptiveInstanceNorm(out_channels, z_dimension)

        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.noise2 = NoiseInjection(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
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


class Generator(nn.Module):

    def __init__(self, initial_size, z_dimension=128, upsample=6, depth=16):
        """
        Arguments:
            initial_size: a tuple of integers (h, w).
            z_dimension: an integer.
            upsample: an integer.
            depth: an integer.
        """
        super().__init__()

        out_channels = min(depth * (2 ** upsample), 512)
        progression = [InitialGeneratorBlock(initial_size, out_channels, z_dimension)]
        to_rgb = [Conv2d(out_channels, 3, 1)]

        for i in range(upsample):

            m = 2 ** (upsample - i - 1)  # multiplier
            in_channels = min(depth * m * 2, 512)
            out_channels = min(depth * m, 512)

            block = GeneratorBlock(in_channels, out_channels, z_dimension)
            converter = Conv2d(out_channels, 3, 1)

            progression.append(block)
            to_rgb.append(converter)

        """
        If upsample = 6 then tuples
        (i, output shape of a block) are:
        0, [b, 32 * depth, 2 * h, 2 * w]
        1, [b, 16 * depth, 4 * h, 4 * w]
        2, [b, 8 * depth, 8 * h, 8 * w]
        3, [b, 4 * depth, 16 * h, 16 * w]
        4, [b, 2 * depth, 32 * h, 32 * w]
        5, [b, depth, 64 * h, 64 * w]
        """

        self.progression = nn.ModuleList(progression)
        self.to_rgb = nn.ModuleList(to_rgb)
        self.upsample = upsample

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, z_dimension].
        Returns:
            a list with float tensors. Where `i`-th tensor
            has shape [b, 3, s * h, s * w] with s = 2 ** i.
            Integer `i` is in range [0, upsample].
        """

        x = self.progression[0](z)
        # it has spatial size (h, w)

        outputs = []
        outputs.append(self.to_rgb[0](x))

        for i in range(self.upsample):

            x = self.progression[i + 1](x, z)
            outputs.append(self.to_rgb[i + 1](x))

        return outputs


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.layers(x)


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


class FinalDiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, initial_size):
        super().__init__()

        h, w = initial_size
        # h and w are small

        self.layers = nn.Sequential(
            MinibatchStdDev(),
            Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=(h, w)),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels, 1, 1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b].
        """
        b = x.size(0)
        return self.layers(x).view(b)


class Discriminator(nn.Module):

    def __init__(self, initial_size, upsample=6, depth=16):
        """
        Arguments:
            initial_size: a tuple of integers (h, w).
            upsample: an integer.
            depth: a integer.
        """
        super().__init__()

        from_rgb = [Conv2d(3, depth, 1)]
        progression = [DiscriminatorBlock(depth, 2 * depth)]

        for i in range(1, upsample):

            m = 2 ** i  # multiplier
            in_channels = min(depth * m, 512)
            out_channels = min(depth * m * 2, 512)

            from_rgb.append(Conv2d(3, depth, 1))
            progression.append(DiscriminatorBlock(in_channels + depth, out_channels))

        """
        If upsample = 6 then
        tuples (i, output shape of a block) are:
        1, [b, 4 * depth, 16 * h, 16 * w]
        2, [b, 8 * depth, 8 * h, 8 * w]
        3, [b, 16 * depth, 4 * h, 4 * w]
        4, [b, 32 * depth, 2 * h, 2 * w]
        5, [b, 64 * depth, h, w]
        """

        self.final_from_rgb = Conv2d(3, depth, 1)
        self.progression = nn.ModuleList(progression)
        self.from_rgb = nn.ModuleList(from_rgb)
        self.upsample = upsample

    def forward(self, inputs):
        """
        Arguments:
            inputs: a list with float tensors. Where `i`-th tensor
            has shape [b, 3, s * h, s * h] with s = 2 ** i.
            Integer `i` is in range [0, upsample].
        Returns:
            a float tensor with shape [b, 16 + 512, h, w].
        """
        upsample = self.upsample

        x = inputs[upsample]
        x = self.from_rgb[0](x)
        # it has shape [b, depth, s * h, s * w],
        # where s = 2 ** depth

        x = self.progression[0](x)
        # it has shape [b, 2 * depth, s * h / 2, s * w / 2]

        for i in range(1, upsample):

            f = self.from_rgb[i](inputs[upsample - i])
            x = torch.cat([x, f], dim=1)

            # x has spatial size (s * h, s * w),
            # where s = 2 ** (upsample - i)

            x = self.progression[i](x)

        f = self.final_from_rgb(inputs[0])
        x = torch.cat([x, f], dim=1)
        # it has spatial size (h, w)

        return x
