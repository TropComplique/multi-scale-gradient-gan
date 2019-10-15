import torch
from torch import optim
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import math
import os
import copy
from PIL import Image, ImageFilter
from tqdm import tqdm
from networks import Generator, Discriminator
from networks import FinalDiscriminatorBlock

from torch.backends import cudnn
cudnn.benchmark = True


DEVICES = [torch.device('cuda:0'), torch.device('cuda:1')]
WIDTH, HEIGHT = 512, 768
UPSAMPLE = 7  # number of upsamplings

BATCH_SIZE = 20  # must be divisible by 2 * num_gpus
NUM_ITERATIONS = 100000  # 439 iterations in one epoch
IMAGES_PATH = '/home/dan/datasets/feidegger/images/'  # it contains 8792 images
LOGS_DIR = 'summaries/run00/'
PLOT_IMAGE_STEP = 300
MODELS_DIR = 'checkpoints/'
SAVE_EPOCH = 10


def train():

    epoch = 1
    progress_bar = tqdm(range(NUM_ITERATIONS + 1))
    writer = SummaryWriter(LOGS_DIR)

    dataset = Images(folder=IMAGES_PATH, size=(HEIGHT, WIDTH))
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8, drop_last=True)
    data_loader = iter(loader)

    depth = 16
    z_dimension = 128

    s = 2 ** UPSAMPLE
    assert WIDTH % s == 0 and HEIGHT % s == 0
    initial_size = (HEIGHT // s, WIDTH // s)

    generator = Generator(z_dimension, initial_size, UPSAMPLE, depth).to(DEVICES[0])
    discriminator = Discriminator(initial_size, UPSAMPLE, depth).to(DEVICES[0])

    score_predictor = FinalDiscriminatorBlock(discriminator.out_channels, initial_size).to(DEVICES[0])
    # this block is separated because it contains MinibatchStdDev (it doesn't parallelize)

    generator = nn.DataParallel(generator, device_ids=DEVICES)
    discriminator = nn.DataParallel(discriminator, device_ids=DEVICES)
    discriminator = nn.Sequential(discriminator, score_predictor)

    g_optimizer = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.0, 0.99))

    # exponential moving average of weights
    generator_ema = copy.deepcopy(generator)

    def get_noise(b):
        z = torch.randn(b, z_dimension, device=DEVICES[0])
        z = z / z.norm(p=2, dim=1, keepdim=True)
        return z

    # for visualization in tensorboard
    noise_vectors = torch.split(get_noise(10), 1)

    for i in progress_bar:

        try:
            images = next(data_loader)
            # it has shape [b, 3, HEIGHT, WIDTH]

        except (OSError, StopIteration):

            if epoch % SAVE_EPOCH == 0:
                state = {
                    'generator': generator.module.state_dict(),
                    'generator_ema': generator_ema.module.state_dict()
                }
                save_path = os.path.join(MODELS_DIR, f'train_epoch_{epoch}.model')
                torch.save(state, save_path)

            # start a new epoch
            data_loader = iter(loader)
            images = next(data_loader)
            epoch += 1

        b = images.size(0)  # batch size
        images = images.to(DEVICES[0])
        images.requires_grad = True

        downsampled = [images]
        for _ in range(UPSAMPLE):
            images = F.avg_pool2d(images, 2)
            downsampled.append(images)

        # from lowest to biggest resolution
        real_images = downsampled[::-1]

        z = get_noise(b)  # shape [b, z_dimension]
        fake_images = generator(z)
        fake_images_detached = [x.detach() for x in fake_images]

        real_scores = discriminator(real_images)
        fake_scores = discriminator(fake_images_detached)
        # they have shape [b/2] (because of pacgan)

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        discriminator_loss = F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()

        g = grad(real_scores.sum(), images, create_graph=True)[0]
        # it has shape [b, 3, h, w]

        R1 = 0.5 * g.view(b, -1).pow(2).sum(1).mean(0)
        # it has shape []

        d_optimizer.zero_grad()
        (discriminator_loss + R1).backward()
        d_optimizer.step()

        discriminator.requires_grad_(False)
        real_images_detached = [x.detach() for x in real_images]

        real_scores = discriminator(real_images_detached)
        fake_scores = discriminator(fake_images)
        # they have shape [b/2] (because of pacgan)

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        generator_loss = F.relu(1.0 + r).mean() + F.relu(1.0 - f).mean()

        g_optimizer.zero_grad()
        generator_loss.backward()
        g_optimizer.step()

        discriminator.requires_grad_(True)
        accumulate(generator_ema, generator)

        g = generator_loss.item()
        d = discriminator_loss.item()
        r = R1.item()

        writer.add_scalar('losses/generator_loss', g, i)
        writer.add_scalar('losses/discriminator_loss', d, i)
        writer.add_scalar('losses/R1_penalty', r, i)

        if i % PLOT_IMAGE_STEP == 0:

            for j, z in enumerate(noise_vectors):
                with torch.no_grad():
                    images = generator(z)
                    for k, x in enumerate(images):
                        x = 0.5 * (x + 1.0)
                        x = x[0].cpu()
                        writer.add_image(f'sample_{j}/scale_{k}', x, i)

        description = f'epoch: {epoch}'
        progress_bar.set_description(description)


def accumulate(model_accumulator, model, decay=0.993):

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)


class Images(Dataset):

    def __init__(self, folder, size):
        """
        Arguments:
            folder: a string, the path to a folder with images.
            size: a tuple of integers (h, w).
        """

        self.names = os.listdir(folder)
        self.folder = folder

        color_transforms = [
            transforms.Lambda(lambda x: x.filter(ImageFilter.MedianFilter(3))),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=2.0, hue=0.5),
            transforms.RandomGrayscale(p=0.3)
        ]

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomChoice(color_transforms),
            transforms.RandomHorizontalFlip(),
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
