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


Z_DIMENSION = 512
IMAGE_SIZE = 256
DEPTH = int(math.log2(IMAGE_SIZE)) - 2
BATCH_SIZE = 20
NUM_ITERATIONS = 300000
DEVICE = torch.device('cuda:0')


def accumulate(model_accumulator, model, decay=0.999):

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)


class Images(Dataset):

    def __init__(self, folder):
        """
        Arguments:
            folder: a string, the path to a folder with images.
        """
        self.names = os.listdir(folder)
        self.folder = folder
        
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        """
        """
        name = self.names[i]
        path = os.path.join(self.folder, name)
        image = Image.open(path).convert('RGB')
        #w, h = image.size
        #assert w > 256 and h > 256, f'size: {w}, {h}'
        return self.transform(image)
    
dataset = Images(folder='/home/dan/work/feidegger/patterns/images/')
generator = Generator(DEPTH, Z_DIMENSION).to(DEVICE)
discriminator = Discriminator(DEPTH).to(DEVICE)

g_optimizer = optim.Adam(generator.parameters(), lr=3e-3, betas=(0.0, 0.99))
d_optimizer = optim.Adam(discriminator.parameters(), lr=3e-3, betas=(0.0, 0.99))

generator_ema = Generator(DEPTH, Z_DIMENSION).to(DEVICE)
accumulate(generator_ema, generator, 0.0)


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def get_data_loader():
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
    return loader


def train():

    epoch = 1
    progress_bar = tqdm(range(NUM_ITERATIONS))
    
    loader = get_data_loader()
    data_loader = iter(loader)

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    for i in progress_bar:

        try:
            images = next(data_loader)
            # it has shape [b, 3, h, w],
            # where h = w = IMAGE_SIZE

        except (OSError, StopIteration):

            state = {
                'generator': generator.module.state_dict(),
                'discriminator': discriminator.module.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict()
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
        #with torch.no_grad():
        fake_images = generator(z)
        #fake_images = [x.detach() for x in fake_images]
        
        real_scores = discriminator(images)
        fake_scores = discriminator(fake_images)
        # they have shape [b]

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        discriminator_loss = F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()

        discriminator.zero_grad()
        discriminator_loss.backward()
        d_optimizer.step()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_images = generator(z)
        fake_scores = discriminator(fake_images)
        
        real_scores = real_scores.detach()
        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        generator_loss = F.relu(1.0 + r).mean() + F.relu(1.0 - f).mean()

        generator.zero_grad()
        generator_loss.backward()
        g_optimizer.step()

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        accumulate(generator_ema, generator)

        description = f'epoch: {epoch}, generator: {generator_loss.item():.3f}, discriminator: {discriminator_loss.item():.3f}'
        progress_bar.set_description(description)


train()
