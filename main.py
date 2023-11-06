import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from tqdm import tqdm

from PIL import Image
import wandb


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # initialize all weights with Isotropic gaussian (µ = 0, σ = 0.02), bias with Constant(0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


class RandomMatrix:
    def __init__(self, dataset):
        # calculate FFT of the datapoints
        n = 1000
        datapoints = []

        for x, y in dataset:
            datapoints.append(x)

            if len(datapoints) > n:
                break
        datapoints = torch.stack(datapoints)
        data_fft2 = torch.fft.fft2(datapoints)
        fft_flat = data_fft2.flatten(2).contiguous()
        fft_flat_mean, fft_flat_std = fft_flat.mean(dim=0), fft_flat.std(dim=0)
        self.shape = dataset[0][0].shape

        self.fft_mean = fft_flat_mean.view(self.shape)
        self.fft_std = fft_flat_std.view(self.shape)

    def __call__(self, seed=None, batch_size=1):
        if seed is None:
            x = torch.randn((batch_size, *self.shape))
        else:
            x = torch.randn(
                (batch_size, *self.shape), generator=torch.Generator().manual_seed(seed)
            )

        x = torch.fft.ifft2(x * self.fft_std + self.fft_mean).real

        return x


def train(
    batch_size=256,
    lr=1e-4,
    validate_step=100,
    epochs=1000,
    tight_lambda=0.1,
    copy_interval=1,
):
    wandb.init(
        project="IGN",
        entity="simo",
        config={
            "batch_size": batch_size,
            "lr": lr,
            "validate_step": validate_step,
            "epochs": epochs,
            "tight_lambda": tight_lambda,
        },
        name="fft",
    )

    # prepare data
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((64, 64)),
        ]
    )
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # random matrix generator, because the method does not work with "purely random gaussian matrix"
    rmg = RandomMatrix(dataset)

    device = "cuda:0"

    model = Autoencoder()
    model.to(device)
    model.train()
    optimizer = Adam(
        model.parameters(),
        lr=lr,
    )

    model_copy = Autoencoder()
    model_copy.to(device)

    model_copy.requires_grad_(False)

    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(dataloader)

        for x, _ in pbar:
            # copy state
            optimizer.zero_grad()

            if global_step % copy_interval == 0:
                model_copy.load_state_dict(model.state_dict())

            if global_step % validate_step == 0:
                # validate model by sampling
                with torch.no_grad():
                    z = rmg(None, batch_size=25)
                    z = z.to(device)
                    model.eval()
                    fz = model(z)
                    print(fz.shape)
                    model.train()
                    fz_grid = make_grid(fz, nrow=5, normalize=True, value_range=(-1, 1))
                    save_image(fz_grid, f"./contents/ign_{global_step}.png")

                    fz_grid = fz_grid.cpu()
                    fz_grid = (fz_grid * 255).byte()
                    fz_grid = fz_grid.permute(1, 2, 0).numpy()
                    fz_grid = Image.fromarray(fz_grid)

                    z_grid = make_grid(z, nrow=5, normalize=True, value_range=(-1, 1))
                    z_grid = z_grid.cpu()
                    z_grid = (z_grid * 255).byte()
                    z_grid = z_grid.permute(1, 2, 0).numpy()
                    z_grid = Image.fromarray(z_grid)

                    wandb.log({"sample": [wandb.Image(fz_grid)]})
                    wandb.log({"z": [wandb.Image(z_grid)]})

            x = x.to(device)
            z = rmg(None, batch_size=batch_size)
            z = z.to(device)

            fx = model(x)
            loss_rec = (fx - x).abs().mean()
            loss_rec.backward()

            fz = model(z)
            f_fz = model_copy(fz)
            loss_idem = (f_fz - fz).abs().mean()
            loss_idem.backward()

            f_z = fz.detach()
            ff_z = model(f_z)
            loss_tight = -(ff_z - f_z).abs().reshape(batch_size, -1).mean(1)
            with torch.no_grad():
                l_rec_z = (f_z - z).abs().reshape(batch_size, -1).mean(1) * 1.5

            loss_tight = torch.tanh(loss_tight / l_rec_z) * l_rec_z * tight_lambda
            loss_tight = loss_tight.mean()

            loss_tight.backward()

            pbar.set_description(
                f"loss_rec: {loss_rec.item():.4f}, loss_idem: {loss_idem.item():.4f}, loss_tight: {loss_tight.item():.4f}"
            )

            wandb.log(
                {
                    "loss_rec": loss_rec.item(),
                    "loss_idem": loss_idem.item(),
                    "loss_tight": loss_tight.item(),
                    "overall_loss": loss_rec.item()
                    + loss_idem.item()
                    + loss_tight.item(),
                }
            )

            optimizer.step()
            global_step += 1


if __name__ == "__main__":
    train()
