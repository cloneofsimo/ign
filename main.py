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


def unet():
    from diffusers import UNet2DConditionModel

    # config referenced from https://huggingface.co/nota-ai/bk-sdm-small/blob/main/unet/config.json
    unet = UNet2DConditionModel.from_config(
        {
            "act_fn": "silu",
            "attention_head_dim": 8,
            "block_out_channels": [32, 64, 128, 128],
            "center_input_sample": False,
            "class_embed_type": None,
            "class_embeddings_concat": False,
            "conv_in_kernel": 3,
            "conv_out_kernel": 3,
            "cross_attention_dim": 768,
            "cross_attention_norm": None,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ],
            "downsample_padding": 1,
            "dual_cross_attention": False,
            "encoder_hid_dim": None,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 1,
            "mid_block_only_cross_attention": None,
            "mid_block_scale_factor": 1,
            "mid_block_type": None,
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "out_channels": 3,
            "projection_class_embeddings_input_dim": None,
            "resnet_out_scale_factor": 1.0,
            "resnet_skip_time_act": False,
            "resnet_time_scale_shift": "default",
            "sample_size": 64,
            "time_cond_proj_dim": None,
            "time_embedding_act_fn": None,
            "time_embedding_type": "positional",
            "timestep_post_act": None,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ],
            "upcast_attention": False,
            "use_linear_projection": False,
        }
    )

    return unet


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


def dummy_unet():
    # https://github.com/cloneofsimo/minDiffusion/blob/master/superminddpm.py
    return DummyEpsModel(3)


class RandomMatrix:
    def __init__(self, dataset):
        # calculate FFT of the datapoints
        n = 10000
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
    validate_step=50,
    epochs=1000,
    tight_lambda=1,
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
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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

    model = dummy_unet()
    model.to(device)
    model.train()
    optimizer = Adam(
        model.parameters(),
        lr=lr,
    )

    model_copy = dummy_unet()
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
                    z = rmg(None, batch_size=32)
                    z = z.to(device)
                    model.eval()
                    fz = model(z)
                    print(fz.shape)
                    model.train()
                    grid = make_grid(fz, nrow=4, normalize=True, value_range=(-1, 1))
                    save_image(grid, f"./contents/ign_{global_step}.png")

                    grid = grid.cpu()
                    grid = (grid * 255).byte()
                    grid = grid.permute(1, 2, 0).numpy()
                    grid = Image.fromarray(grid)
                    wandb.log({"sample": [wandb.Image(grid)]})

            x = x.to(device)
            z = rmg(None, batch_size=batch_size)
            z = z.to(device)

            fx = model(x)
            loss_rec = (fx - x).pow(2).mean()
            loss_rec.backward()

            fz = model(z)
            f_fz = model_copy(fz)
            loss_idem = (f_fz - fz).pow(2).mean()
            loss_idem.backward()

            f_z = fz.detach()
            ff_z = model(f_z)
            loss_tight = -(ff_z - f_z).pow(2).mean() * tight_lambda
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
                    + loss_tight.item() * tight_lambda,
                }
            )

            optimizer.step()
            global_step += 1


if __name__ == "__main__":
    train()
