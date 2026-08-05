import os
import argparse
import copy
import math
import torch
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, AutoencoderKL
from diffusers.training_utils import EMAModel
import wandb


# ============================================================
# Shared scheduler config — single source of truth
# ============================================================
SCHEDULER_KWARGS = dict(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",   # cosine schedule: more stable than scaled_linear
    prediction_type="epsilon",
    clip_sample=False,
)

EVAL_SCHEDULER_KWARGS = dict(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="epsilon",
    clip_sample=True,           # clamp pred_x0 each step ...
    clip_sample_range=4.0,      # ... to +-4 sigma (real latents ~N(0,1))
    timestep_spacing="leading", # never start exactly at t=999 to avoid explosion
)

# UNet tuned for 3070 Ti 8GB:
# - (128,256,384,512) instead of (192,384,576,768) — ~40% fewer params, fits in VRAM
# - attention only at 8x8 and 4x4 — avoids 32x32 attention memory explosion
# - resnet_time_scale_shift="scale_shift" kept — better time conditioning
UNET_KWARGS = dict(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 384, 512),
    attention_head_dim=8,
    resnet_time_scale_shift="scale_shift",
    down_block_types=(
        "DownBlock2D",        # 32x32 — no attention
        "DownBlock2D",        # 16x16 — no attention
        "AttnDownBlock2D",    # 8x8   — attention here
        "AttnDownBlock2D",    # 4x4   — and here
    ),
    up_block_types=(
        "AttnUpBlock2D",      # 4x4
        "AttnUpBlock2D",      # 8x8
        "UpBlock2D",          # 16x16
        "UpBlock2D",          # 32x32
    ),
)


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted([
            p for p in self.image_dir.glob("*.png") if p.is_file()
        ])
        if not self.image_paths:
            self.image_paths = sorted([
                p for p in self.image_dir.glob("*.jpg") if p.is_file()
            ])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.02, contrast=0.02, saturation=0.02, hue=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx]).convert("RGB"))


@torch.no_grad()
def generate_and_save_images(unet, vae, step, device, save_folder, num_samples=8):
    """Fast eval using DDIM 50 steps with numerically stable settings."""
    unet.eval()
    eval_scheduler = DDIMScheduler(**EVAL_SCHEDULER_KWARGS)
    eval_scheduler.set_timesteps(50)

    generator = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn((num_samples, 4, 32, 32), device=device, generator=generator)

    for t in eval_scheduler.timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = unet(latents, t_batch).sample
        latents = eval_scheduler.step(noise_pred, t, latents).prev_sample

    # Health check
    lat_std = latents.std().item()
    if not (0.5 <= lat_std <= 2.0):
        print(f"[eval][WARN] final latent std={lat_std:.2f} (expected ~1.0)")

    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    images = (images.clamp(-1, 1) + 1) / 2

    step_folder = os.path.join(save_folder, f"step_{step}")
    os.makedirs(step_folder, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for i, img in enumerate(images):
        to_pil(img.cpu()).save(os.path.join(step_folder, f"{i:04d}.png"))
    print(f"[Step {step}] Saved {num_samples} eval samples to {step_folder}")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to unet checkpoint folder to resume from.")
    args = parser.parse_args()

    # ========= Hyperparameters (tuned for 3070 Ti 8GB) =========
    train_epochs               = 150
    batch_size                 = 8    # 16 is too large for 8GB with this UNet
    gradient_accumulation_steps = 2   # effective batch = 16
    lr                         = 1e-4
    warmup_steps               = 500
    ema_decay                  = 0.9999
    eval_freq                  = 2000
    save_freq                  = 5000
    image_dir                  = "images_aligned"
    run_id                     = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir                 = os.path.join("outputs", run_id)
    device                     = "cuda" if torch.cuda.is_available() else "cpu"

    sample_folder = os.path.join(output_dir, "samples")
    ckpt_folder   = os.path.join(output_dir, "checkpoints")
    os.makedirs(sample_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)

    # ========== VAE (frozen) ==========
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    vae.eval()

    # ========== UNet ==========
    unet = UNet2DModel(**UNET_KWARGS).to(device)

    # Shape sanity check
    with torch.no_grad():
        _x = torch.randn(1, 4, 32, 32, device=device)
        _t = torch.zeros(1, dtype=torch.long, device=device)
        assert unet(_x, _t).sample.shape == _x.shape, "UNet shape mismatch"
        del _x, _t
    print("UNet shape check passed.")

    unet.train()

    # EMAModel from diffusers — more reliable than manual implementation
    ema_model = EMAModel(unet.parameters(), decay=ema_decay)

    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
    )

    # ========== Dataset ==========
    dataset = ImageDataset(image_dir, image_size=256)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # ========== LR Schedule ==========
    total_steps = train_epochs * math.ceil(len(dataloader) / gradient_accumulation_steps)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========== Schedulers ==========
    noise_scheduler = DDPMScheduler(**SCHEDULER_KWARGS)

    # ========== W&B ==========
    wandb.init(project="hw5-professor-diffusion", config={
        "train_epochs": train_epochs,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "lr": lr, "warmup_steps": warmup_steps, "ema_decay": ema_decay,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
        "block_out_channels": str(UNET_KWARGS["block_out_channels"]),
        "attention": "8x8_and_4x4_only",
        "run_id": run_id,
        "resume_from": args.resume_from,
    })

    # ========== Resume ==========
    start_step, start_epoch = 0, 0
    if args.resume_from is not None:
        unet_ckpt = UNet2DModel.from_pretrained(args.resume_from).to(device)
        unet.load_state_dict(unet_ckpt.state_dict())
        del unet_ckpt
        ckpt_dir = os.path.dirname(args.resume_from)
        step_num = os.path.basename(args.resume_from).replace("unet_", "")
        state_path = os.path.join(ckpt_dir, f"training_state_{step_num}.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            if state.get("rng_state") is not None and torch.cuda.is_available():
                rng = state["rng_state"]
                torch.cuda.set_rng_state(
                    rng if isinstance(rng, torch.ByteTensor) else rng.cpu().byte()
                )
            start_step, start_epoch = state["step"], state["epoch"]
            print(f"Resumed from step {start_step}, epoch {start_epoch}")
        else:
            print(f"[warn] No training_state found, resuming weights only.")

    # ========== Training Loop ==========
    step = start_step
    loss_accum = 0.0
    optimizer.zero_grad()
    loss_window = []
    pbar = tqdm(total=(train_epochs - start_epoch) * len(dataloader), desc="Training")

    for epoch in range(start_epoch, train_epochs):
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
            noise_pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            # Timestep-grouped loss for wandb monitoring
            with torch.no_grad():
                psl = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2, 3))
                em, mm, lm = timesteps < 333, (timesteps >= 333) & (timesteps < 667), timesteps >= 667
                le   = psl[em].mean().item() if em.any()  else None
                lmid = psl[mm].mean().item() if mm.any()  else None
                ll   = psl[lm].mean().item() if lm.any()  else None

            loss_scalar = loss.item()
            (loss / gradient_accumulation_steps).backward()
            loss_accum += loss_scalar

            if step % gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0).item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema_model.step(unet.parameters())

                avg_loss = loss_accum / gradient_accumulation_steps
                log = {
                    "train/loss": avg_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm,
                    "train/epoch": epoch, "train/step": step,
                }
                if le is not None:   log["train/loss_early"] = le
                if lmid is not None: log["train/loss_mid"]   = lmid
                if ll is not None:   log["train/loss_late"]  = ll
                wandb.log(log, step=step)
                loss_accum = 0.0

            loss_window.append(loss_scalar)
            if len(loss_window) > 100:
                loss_window.pop(0)
            pbar.set_postfix({"loss": f"{sum(loss_window)/len(loss_window):.4f}", "ep": epoch})
            pbar.update(1)

            if step % eval_freq == 0:
                # Eval with EMA weights
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())
                generate_and_save_images(unet, vae, step, device, sample_folder)
                ema_model.restore(unet.parameters())
                unet.train()

            if step % save_freq == 0:
                # Save EMA weights for inference
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())
                unet.save_pretrained(os.path.join(ckpt_folder, f"unet_{step}"))
                ema_model.restore(unet.parameters())
                torch.save({
                    "step": step, "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                }, os.path.join(ckpt_folder, f"training_state_{step}.pt"))
                print(f"[Step {step}] Checkpoint saved.")

    # Final save with EMA weights
    ema_model.store(unet.parameters())
    ema_model.copy_to(unet.parameters())
    unet.save_pretrained(os.path.join(ckpt_folder, "unet_final"))
    ema_model.restore(unet.parameters())
    torch.save({
        "step": step, "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, os.path.join(ckpt_folder, "training_state_final.pt"))
    print("Training complete. Weights saved to unet_final.")
    wandb.finish()


def overfit_test(num_steps: int = 300, log_every: int = 50):
    """Single-batch overfit sanity check. Expects loss < 0.1 within ~300 steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    unet = UNet2DModel(**UNET_KWARGS).to(device)
    noise_scheduler = DDPMScheduler(**SCHEDULER_KWARGS)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-4, weight_decay=0.0)

    dataset = ImageDataset("images_aligned", image_size=256)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True)
    fixed_batch = next(iter(loader)).to(device)

    with torch.no_grad():
        fixed_latents = vae.encode(fixed_batch).latent_dist.sample()
        fixed_latents = fixed_latents * vae.config.scaling_factor

    print(f"Overfit test: {num_steps} steps on {fixed_latents.shape[0]} images")
    unet.train()
    recent = []

    for step in range(1, num_steps + 1):
        noise = torch.randn_like(fixed_latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (fixed_latents.shape[0],), device=device,
        ).long()
        noisy = noise_scheduler.add_noise(fixed_latents, noise, timesteps)
        loss = F.mse_loss(unet(noisy, timesteps).sample, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        recent.append(loss.item())
        if len(recent) > 50: recent.pop(0)

        if step % log_every == 0 or step == 1:
            avg = sum(recent) / len(recent)
            print(f"  step {step:4d} | loss {loss.item():.5f}  (avg-50: {avg:.5f})")

    final  = sum(recent) / len(recent)
    status = "PASS" if final < 0.1 else "FAIL"
    print(f"\n[{status}] avg last-50 loss = {final:.5f}  (expect < 0.1)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit", action="store_true",
                        help="Run single-batch overfit test instead of full training")
    parser.add_argument("--overfit_steps", type=int, default=300)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    if args.overfit:
        overfit_test(num_steps=args.overfit_steps)
    else:
        train()