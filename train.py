import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import math
import os
import copy
from PIL import Image
from tqdm import tqdm
from networks import Generator, Discriminator, FinalDiscriminatorBlock

from torch.backends import cudnn
cudnn.benchmark = True

WIDTH, HEIGHT = 256, 384
UPSAMPLE = 6  # number of upsamplings

BATCH_SIZE = 64
NUM_ITERATIONS = 92000
DEVICE = torch.device('cuda:0')
IMAGES_PATH = '/home/dan/datasets/feidegger/images/'
LOGS_DIR = 'summaries/'
MODELS_DIR = 'checkpoints/'
PLOT_IMAGE_STEP = 200


def train():

    epoch = 1
    progress_bar = tqdm(range(NUM_ITERATIONS))
    writer = SummaryWriter(LOGS_DIR)

    dataset = Images(folder=IMAGES_PATH, size=(HEIGHT, WIDTH))
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8, drop_last=True)
    data_loader = iter(loader)

    z_dimension = 128
    s = 2 ** UPSAMPLE
    assert WIDTH % s == 0 and HEIGHT % s == 0
    initial_size = (HEIGHT // s, WIDTH // s)

    DEVICES = [torch.device('cuda:0'), torch.device('cuda:1')]

    generator = Generator(initial_size, z_dimension, upsample=UPSAMPLE).to(DEVICES[0])
    discriminator = Discriminator(initial_size, upsample=UPSAMPLE).to(DEVICES[0])

    # this block is separated because it contains MinibatchStdDev
    final_block = FinalDiscriminatorBlock(16 + 512, initial_size).to(DEVICES[0])

    generator = nn.DataParallel(generator, device_ids=DEVICES)
    discriminator = nn.DataParallel(discriminator, device_ids=DEVICES)

    g_optimizer = optim.Adam(generator.parameters(), lr=3e-3, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=3e-3, betas=(0.0, 0.99))

    generator_ema = copy.deepcopy(generator)

    noise_vectors = []
    for _ in range(10):
        z = torch.randn(1, z_dimension, device=DEVICE)
        z = z / z.norm(p=2, dim=1, keepdim=True)
        noise_vectors.append(z)

    for i in progress_bar:

        try:
            images = next(data_loader)
            # it has shape [b, 3, h, w],
            # where h = w = IMAGE_SIZE

        except (OSError, StopIteration):

            # save every fifth epoch
            if epoch % 25 == 0:
                state = {'generator': generator.state_dict(), 'generator_ema': generator_ema.state_dict()}
                save_path = os.path.join(MODELS_DIR, f'train_epoch_{epoch}.model')
                torch.save(state, save_path)

            # start a new epoch
            data_loader = iter(loader)
            images = next(data_loader)
            epoch += 1

        b = images.shape[0]
        images = images.to(DEVICE)

        downsampled = [images]
        for _ in range(UPSAMPLE):
            images = F.avg_pool2d(images, 2)
            downsampled.append(images)

        # from lowest to biggest resolution
        images = downsampled[::-1]

        z = torch.randn(b, z_dimension, device=DEVICE)
        z = z / z.norm(p=2, dim=1, keepdim=True)

        fake_images = generator(z)
        fake_images_detached = [x.detach() for x in fake_images]

        real_scores = discriminator(images)
        fake_scores = discriminator(fake_images_detached)
        # they have shape [b]

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        discriminator_loss = F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()

        d_optimizer.zero_grad()
        discriminator_loss.backward()
        d_optimizer.step()

        discriminator.requires_grad_(False)

        real_scores = discriminator(images)
        fake_scores = discriminator(fake_images)
        # they have shape [b]

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

        writer.add_scalar('losses/generator_loss', g, i)
        writer.add_scalar('losses/discriminator_loss', d, i)

        if i % PLOT_IMAGE_STEP == 0:

            for j, z in enumerate(noise_vectors):
                with torch.no_grad():
                    xs = generator(z)
                    for k, x in enumerate(xs):
                        x = 0.5 * (x + 1.0)
                        x = x.clamp(0.0, 1.0).cpu()
                        writer.add_image(f'sample_{j}/scale_{k}', x[0], i)

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

        self.transform = transforms.Compose([
            transforms.Resize(size),
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
