import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, AutoencoderKL
from diffusers.training_utils import EMAModel
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F


# ============================================================
# Shared scheduler config — single source of truth
# Both train() and overfit_test() read from here so they are
# always consistent.
# ============================================================
SCHEDULER_KWARGS = dict(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="epsilon",
    clip_sample=False,
)

UNET_KWARGS = dict(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(192, 384, 576, 768),
    attention_head_dim=8,
    resnet_time_scale_shift="scale_shift",
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
)


class ImageDataset(Dataset):
    """Dataset for loading images from a directory."""

    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = sorted(
            [p for p in self.image_dir.glob("*.png") if p.is_file()]
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.02,
                contrast=0.02,
                saturation=0.02,
                hue=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def build_weighted_dataloader(
    orig_dir: str,
    extra_dir: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    orig_weight: float = 1.0,
    extra_weight: float = 0.4,
) -> DataLoader:
    orig_ds = ImageDataset(orig_dir, image_size=image_size)
    datasets = [orig_ds]
    weights = [orig_weight] * len(orig_ds)

    if extra_dir and Path(extra_dir).exists():
        extra_ds = ImageDataset(extra_dir, image_size=image_size)
        if len(extra_ds) > 0:
            datasets.append(extra_ds)
            weights += [extra_weight] * len(extra_ds)
            n_orig, n_extra = len(orig_ds), len(extra_ds)
            eff_extra = (n_extra * extra_weight) / (
                n_orig * orig_weight + n_extra * extra_weight
            )
            print(
                f"Extra data: {n_extra} imgs  "
                f"effective extra ratio ≈ {eff_extra:.1%}"
            )
        else:
            print(f"Warning: {extra_dir} is empty, skipping.")
    else:
        print(f"No extra data at '{extra_dir}', training on original only.")

    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(combined),
        replacement=True,
    )
    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )


@torch.no_grad()
def generate_and_save_images(unet, vae, eval_scheduler, step, device, save_folder, use_amp=False):
    unet.eval()
    os.makedirs(save_folder, exist_ok=True)
    to_pil = transforms.ToPILImage()

    latents = torch.randn((16, 4, 32, 32), device=device)
    eval_scheduler.set_timesteps(50)

    for t in eval_scheduler.timesteps:
        with torch.autocast("cuda", enabled=use_amp):
            noise_pred = unet(latents, t).sample
        latents = eval_scheduler.step(noise_pred, t, latents).prev_sample

    # Health check: a correct epsilon/LDM run ends with latent std ~1.0.
    # std in the tens/hundreds => terminal-step explosion (see eval_scheduler note).
    lat_std = latents.std().item()
    if not (0.5 <= lat_std <= 2.0):
        print(f"[eval][WARN] final latent std={lat_std:.2f} (expected ~1.0) "
              f"-> samples will be noise; check sampler config")

    images = vae.decode(latents / vae.config.scaling_factor).sample
    images = (images.clamp(-1, 1) + 1) / 2

    for i, img in enumerate(images):
        to_pil(img.cpu()).save(f"{save_folder}/step{step}_{i:03d}.png")


def train():
    # ========= Hyperparameters ==========
    train_epochs      = 200
    batch_size        = 16
    gradient_accumulation_steps = 1
    lr                = 1e-4
    warmup_steps      = 500
    eval_freq         = 1000
    save_freq         = 10000
    image_dir         = "public_data/images_aligned"
    extra_dir         = "public_data/extra_data_aligned"
    orig_weight       = 1.0
    extra_weight      = 0.4
    output_dir        = "outputs"
    device            = "cuda" if torch.cuda.is_available() else "cpu"

    save_folder = os.path.join(output_dir, "samples")
    os.makedirs(save_folder, exist_ok=True)

    # ========== VAE (frozen) ==========
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    # ========== UNet ==========
    unet = UNet2DModel(**UNET_KWARGS, dropout=0.1).to(device)

    # Shape sanity check
    with torch.no_grad():
        _x = torch.randn(1, 4, 32, 32, device=device)
        _t = torch.zeros(1, dtype=torch.long, device=device)
        _out = unet(_x, _t).sample
        assert _out.shape == _x.shape, \
            f"UNet shape mismatch: got {_out.shape}, expected {_x.shape}"
        del _x, _t, _out
    print("UNet shape check passed.")

    unet.train()

    ema_model = EMAModel(unet.parameters(), decay=0.9999)
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
    )
    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Schedulers (both use shared SCHEDULER_KWARGS) ──────────────────
    noise_scheduler = DDPMScheduler(**SCHEDULER_KWARGS)

    eval_scheduler = DDIMScheduler(
        num_train_timesteps=SCHEDULER_KWARGS["num_train_timesteps"],
        beta_schedule=SCHEDULER_KWARGS["beta_schedule"],
        prediction_type=SCHEDULER_KWARGS["prediction_type"],
        clip_sample=True,          # clamp pred_x0 each step ...
        clip_sample_range=4.0,     # ... to +-4 sigma (real latents ~N(0,1))
        timestep_spacing="leading",
    )
    # ───────────────────────────────────────────────────────────────────

    # ========== Dataset ==========
    dataloader = build_weighted_dataloader(
        orig_dir=image_dir,
        extra_dir=extra_dir,
        image_size=256,
        batch_size=batch_size,
        num_workers=8,
        orig_weight=orig_weight,
        extra_weight=extra_weight,
    )

    total_steps = train_epochs * len(dataloader) // gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ========== Training Loop ==========
    loss_window, step = [], 0
    pbar = tqdm(total=train_epochs * len(dataloader), desc="Training")

    for epoch in range(train_epochs):
        for batch in dataloader:
            step += 1
            pixel_values = batch.to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", enabled=use_amp):
                noise_pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()

            if step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                ema_model.step(unet.parameters())
                optimizer.zero_grad()

            loss_window.append(loss.item())
            if len(loss_window) > 100:
                loss_window.pop(0)
            pbar.set_postfix(loss=f"{sum(loss_window)/len(loss_window):.4f}")
            pbar.update(1)

            if step % eval_freq == 0:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())
                generate_and_save_images(
                    unet, vae, eval_scheduler, step, device, save_folder, use_amp
                )
                ema_model.restore(unet.parameters())
                unet.train()

            if step % save_freq == 0:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())
                unet.save_pretrained(
                    os.path.join(output_dir, "checkpoints", f"unet_{step}")
                )
                ema_model.restore(unet.parameters())

    # Final save with EMA weights
    ema_model.store(unet.parameters())
    ema_model.copy_to(unet.parameters())
    unet.save_pretrained(os.path.join(output_dir, "checkpoints", "unet_final"))
    ema_model.restore(unet.parameters())
    print("Training complete.")


def overfit_test(num_steps: int = 500, log_every: int = 50):
    """Single-batch overfit sanity check.

    Uses the SAME scheduler config as train() (via SCHEDULER_KWARGS).
    Expects loss to drop below 0.05 within ~500 steps if the pipeline
    is correct.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    # dropout=0 so the model can memorise the 4 images
    unet = UNet2DModel(**UNET_KWARGS, dropout=0.0).to(device)

    # ── Same prediction_type as train() ────────────────────────────────
    noise_scheduler = DDPMScheduler(**SCHEDULER_KWARGS)
    # ───────────────────────────────────────────────────────────────────

    # Higher lr + no warmup → faster memorisation
    optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-4, weight_decay=0.0)

    dataset = ImageDataset("public_data/images_aligned", image_size=256)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True)
    fixed_batch = next(iter(loader)).to(device)

    with torch.no_grad():
        fixed_latents = vae.encode(fixed_batch).latent_dist.sample()
        fixed_latents = fixed_latents * vae.config.scaling_factor

    print(f"Overfit test: {num_steps} steps on {fixed_latents.shape[0]} images")
    unet.train()

    recent, tail = [], 50
    for step in range(1, num_steps + 1):
        noise = torch.randn_like(fixed_latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (fixed_latents.shape[0],), device=device,
        ).long()

        noisy = noise_scheduler.add_noise(fixed_latents, noise, timesteps)
        noise_pred = unet(noisy, timesteps).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        recent.append(loss.item())
        if len(recent) > tail:
            recent.pop(0)

        if step % log_every == 0 or step == 1:
            avg = sum(recent) / len(recent)
            print(f"  step {step:4d} | loss {loss.item():.5f}  (avg-{tail} {avg:.5f})")

    final  = sum(recent) / len(recent)
    status = "PASS" if final < 0.1 else "FAIL"
    print(f"\n[{status}] avg last-{tail} loss = {final:.5f}  (expect < 0.1)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overfit", action="store_true",
        help="Run single-batch overfit test instead of full training",
    )
    parser.add_argument("--overfit_steps", type=int, default=500)
    args = parser.parse_args()

    if args.overfit:
        overfit_test(num_steps=args.overfit_steps)
    else:
        train()