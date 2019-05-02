import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from networks import Generator, Discriminator


Z_DIMENSION = 512
IMAGE_SIZE = 256
BATCH_SIZE = 20
NUM_ITERATIONS = 3000000

dataset = None

generator = Generator(code_size).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

g_optimizer = optim.Adam(generator.parameters(), lr=3e-3, betas=(0.0, 0.99))
d_optimizer = optim.Adam(discriminator.parameters(), lr=3e-3, betas=(0.0, 0.99))

generator_ema = Generator(code_size).to(DEVICE)
accumulate(generator_ema, generator, 0.0)


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model_accumulator, model, decay=0.999):

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)


def get_data_loader():

    transform = transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    dataset.transform = transform
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
    return loader


def train():

    epoch = 1
    progress_bar = tqdm(range(NUM_ITERATIONS))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    for i in progress_bar:

        try:
            images = next(data_loader)
            # it has shape [b, 3, h, w],
            # where h = w = resolution

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
        # it has shape [b, 3, h, w],
        # where h = w = resolution

        downsampled = [
            F.avg_pool2d(images, int(np.power(2, i)))
            for i in range(1, DEPTH)
        ]
        images = [images] + downsampled

        z = torch.randn(b, Z_DIMENSION, device=DEVICE)
        fake_images = generator(z)
        # it has shape [b, 3, h, w]

        real_scores = discriminator(images)
        fake_scores = discriminator(fake_images)

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()
        discriminator_loss = F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
        discriminator_loss.backward()
        d_optimizer.step()

        generator.zero_grad()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_images = generator(z)
        fake_scores = discriminator(fake_images)

        r = real_scores - fake_scores.mean()
        f = fake_scores - real_scores.mean()

        generator_loss = F.relu(1.0 + r).mean() + F.relu(1.0 - f).mean()

        generator_loss.backward()
        g_optimizer.step()
        accumulate(g_running, generator.module)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


train()
