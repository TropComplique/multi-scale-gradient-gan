import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import math
import os
from PIL import Image
from tqdm import tqdm
from networks import Generator, Discriminator

from torch.backends import cudnn
cudnn.benchmark = True


Z_DIMENSION = 512
IMAGE_SIZE = 256
DEPTH = int(math.log2(IMAGE_SIZE)) - 2
BATCH_SIZE = 20
NUM_ITERATIONS = 300000
DEVICE = torch.device('cuda:1')


def train():

    epoch = 1
    progress_bar = tqdm(range(NUM_ITERATIONS))

    dataset = Images(folder='/home/dan/work/feidegger/patterns/images/')
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
    data_loader = iter(loader)

    generator = Generator(DEPTH, Z_DIMENSION).to(DEVICE)
    discriminator = Discriminator(DEPTH, max_channels=256).to(DEVICE)

    g_optimizer = optim.Adam(generator.parameters(), lr=3e-3, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=3e-3, betas=(0.0, 0.99))

    generator_ema = Generator(DEPTH, Z_DIMENSION).to(DEVICE)
    accumulate(generator_ema, generator, 0.0)

    for i in progress_bar:

        try:
            images = next(data_loader)
            # it has shape [b, 3, h, w],
            # where h = w = IMAGE_SIZE

        except (OSError, StopIteration):

            state = {
                'generator': generator.state_dict(),
                'generator_ema': generator_ema.state_dict(),
            }
            torch.save(state, f'checkpoints/train_epoch-{epoch}.model')

            # start a new epoch
            data_loader = iter(loader)
            images = next(data_loader)
            epoch += 1

        b = images.shape[0]
        images = images.to(DEVICE)

        downsampled = [
            F.avg_pool2d(images, 2 ** i)
            for i in reversed(range(1, DEPTH + 1))
        ]
        images = downsampled + [images]

        z = torch.randn(b, Z_DIMENSION, device=DEVICE)
        z = (Z_DIMENSION ** 0.5) * z / z.norm(p=2, dim=1, keepdim=True)
        fake_images = generator(z)

        real_scores = discriminator(images)
        fake_scores = discriminator([x.detach() for x in fake_images])
        # they have shape [b]

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        discriminator_loss = F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()

        d_optimizer.zero_grad()
        discriminator_loss.backward()
        d_optimizer.step()

        requires_grad(discriminator, False)

        real_scores = discriminator(images)
        fake_scores = discriminator(fake_images)

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        generator_loss = F.relu(1.0 + r).mean() + F.relu(1.0 - f).mean()

        g_optimizer.zero_grad()
        generator_loss.backward()
        g_optimizer.step()

        requires_grad(discriminator, True)
        accumulate(generator_ema, generator)

        description = f'epoch: {epoch}, generator: {generator_loss.item():.3f}, discriminator: {discriminator_loss.item():.3f}'
        progress_bar.set_description(description)


def accumulate(model_accumulator, model, decay=0.999):

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


class Images(Dataset):

    def __init__(self, folder):
        """
        Arguments:
            folder: a string, the path to a folder with images.
        """

        self.names = os.listdir(folder)
        self.folder = folder

        self.transform = transforms.Compose([
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        """
        Returns:
            a float tensor with shape [3, h, w].
            It represents a RGB image with
            pixel values in [-1, 1] range.
        """

        name = self.names[i]
        path = os.path.join(self.folder, name)
        image = Image.open(path).convert('RGB')

        x = self.transform(image)
        x = 2.0 * x - 1.0
        return x


train()
