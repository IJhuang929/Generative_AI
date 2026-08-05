"""
Kaggle version of train.py.

Checkpoint persistence workflow:
  Session 1: train -> checkpoints saved to /kaggle/working/checkpoints/
             Commit version -> working/ becomes version output
  Session 2: add Session 1's output as input dataset (e.g. "hw5-run1")
             set RESUME_CHECKPOINT below -> resumes from that step

Dataset setup:
  Upload images_aligned.zip as a Kaggle Dataset.
  Kaggle auto-extracts zips, so the image path will be:
    /kaggle/input/<your-dataset-slug>/images_aligned/
  Update IMAGE_DIR below to match your dataset slug.
"""

import os
import sys
import argparse
import json
import torch
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import math
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, AutoencoderKL
import wandb

# =============================================================================
# Kaggle configuration — edit these before running
# =============================================================================

# Path to images_aligned/ inside your uploaded dataset.
# Replace <your-dataset-slug> with your actual Kaggle dataset slug.
IMAGE_DIR = "/kaggle/input/datasets/otto29/hw5-diffusion-images/images_aligned"

# All outputs (checkpoints, samples) go to /kaggle/working/.
# This directory is saved as version output when you commit — download it or
# add it as an input dataset in the next session to resume.
WORKING_DIR = "/kaggle/working"

# To resume from a previous session's checkpoint:
#   1. Add the previous version's output as an input dataset (e.g. slug "hw5-run1").
#   2. Set this to the unet checkpoint path inside that dataset, e.g.:
#      "/kaggle/input/hw5-run1/checkpoints/unet_5000"
#   Leave as None to train from scratch.
RESUME_CHECKPOINT = None

# W&B: set to "offline" to skip internet (logs saved in /kaggle/working/wandb/).
# Set to "online" and add WANDB_API_KEY to Kaggle Secrets (Settings > Secrets)
# to stream logs live.
WANDB_MODE = "offline"

# torch.compile speeds up training but can fail on some Kaggle builds.
# Disable if you see a compile error on startup.
USE_TORCH_COMPILE = False

# =============================================================================


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = sorted([p for p in self.image_dir.glob("*.png") if p.is_file()])
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


@torch.no_grad()
def generate_and_save_images(unet, vae, scheduler, step, device, save_folder, num_samples=8, enable_bar=False, ema_unet=None):
    eval_unet = ema_unet if ema_unet is not None else unet
    eval_unet.eval()

    eval_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="v_prediction",
    )
    eval_scheduler.set_timesteps(50)

    generator = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn((num_samples, 4, 32, 32), device=device, generator=generator)

    timesteps_iter = tqdm(eval_scheduler.timesteps, desc="Sampling") if enable_bar else eval_scheduler.timesteps
    for t in timesteps_iter:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = eval_unet(latents, t_batch).sample
        latents = eval_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents, return_dict=False)[0]
    images = (images.clamp(-1, 1) + 1) / 2

    to_pil = transforms.ToPILImage()
    step_folder = os.path.join(save_folder, f"step_{step:07d}")
    os.makedirs(step_folder, exist_ok=True)
    for i, img in enumerate(images):
        to_pil(img.cpu()).save(os.path.join(step_folder, f"{i:04d}.png"))
    print(f"[Step {step}] Saved {num_samples} sample images to {step_folder}")


def save_checkpoint(ckpt_folder, unet, ema_unet, optimizer, lr_scheduler, step, epoch):
    raw_unet = unet._orig_mod if hasattr(unet, '_orig_mod') else unet

    unet_path = os.path.join(ckpt_folder, f"unet_{step}")
    raw_unet.save_pretrained(unet_path)

    ema_path = os.path.join(ckpt_folder, f"unet_ema_{step}")
    ema_unet.save_pretrained(ema_path)

    state = {
        "step":         step,
        "epoch":        epoch,
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "rng_state":    torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(state, os.path.join(ckpt_folder, f"training_state_{step}.pt"))
    print(f"[Step {step}] Checkpoint saved to {ckpt_folder}")


def load_checkpoint(resume_path, unet, optimizer, lr_scheduler, device):
    unet_ckpt = UNet2DModel.from_pretrained(resume_path).to(device)
    target = unet._orig_mod if hasattr(unet, '_orig_mod') else unet
    target.load_state_dict(unet_ckpt.state_dict())
    del unet_ckpt

    ckpt_folder = os.path.dirname(resume_path)
    step_num = os.path.basename(resume_path).replace("unet_", "")
    state_path = os.path.join(ckpt_folder, f"training_state_{step_num}.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"training_state file not found: {state_path}")

    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    if state["rng_state"] is not None and torch.cuda.is_available():
        rng = state["rng_state"]
        if not isinstance(rng, torch.ByteTensor):
            rng = rng.cpu().byte()
        torch.cuda.set_rng_state(rng)

    start_step  = state["step"]
    start_epoch = state["epoch"]
    print(f"Resumed from step {start_step}, epoch {start_epoch} ({resume_path})")
    return start_step, start_epoch


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=RESUME_CHECKPOINT)
    parser.add_argument("--reset_lr", action="store_true")
    args = parser.parse_args()

    # ========= Hyperparameters ==========
    train_epochs = 200
    batch_size = 8
    gradient_accumulation_steps = 2
    lr = 3e-4
    eval_freq = 1000
    save_freq = 5000
    run_id = datetime.now().strftime("%Y%m%d_%H%M")

    # Checkpoints and samples go directly under /kaggle/working/
    # so they appear as top-level files in the version output and are easy to find.
    ckpt_folder   = os.path.join(WORKING_DIR, "checkpoints")
    sample_folder = os.path.join(WORKING_DIR, "samples", run_id)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(sample_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== W&B ==========
    os.environ.setdefault("WANDB_MODE", WANDB_MODE)
    wandb.init(
        project="hw5-professor-diffusion",
        config={
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr": lr,
            "lr_schedule": "warmup_cosine",
            "eval_freq": eval_freq,
            "eval_scheduler": "ddim_50steps",
            "save_freq": save_freq,
            "unet_block_out_channels": "(256, 512, 512, 512)",
            "unet_layers_per_block": 2,
            "noise_timesteps": 1000,
            "beta_schedule": "scaled_linear",
            "vae": "stabilityai/sd-vae-ft-mse",
            "latent_size": 32,
            "resume_from": args.resume_from,
            "run_id": run_id,
            "min_snr_gamma": 5,
            "prediction_type": "v_prediction",
            "ema_decay": 0.9999,
        }
    )

    print(f"[setup] device={device}", flush=True)

    # ========== VAE ==========
    print("[setup] Loading VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    vae.eval()
    print("[setup] VAE loaded.", flush=True)

    # ========== UNet ==========
    print("[setup] Building UNet...", flush=True)
    unet = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 512, 512),
        down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
    ).to(device)
    unet.train()
    if USE_TORCH_COMPILE:
        unet = torch.compile(unet)
    print("[setup] UNet ready.", flush=True)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== Dataset ==========
    print(f"[setup] Loading dataset from {IMAGE_DIR}...", flush=True)
    dataset = ImageDataset(IMAGE_DIR, image_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    total_steps = train_epochs * math.ceil(math.ceil(len(dataset) / batch_size) / gradient_accumulation_steps)
    warmup_steps = total_steps // 20

    def warmup_cosine_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="v_prediction",
    )

    # ========== EMA ==========
    ema_decay = 0.9999
    ema_unet = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 512, 512),
        down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
    ).to(device)
    raw_unet = unet._orig_mod if hasattr(unet, '_orig_mod') else unet
    ema_unet.load_state_dict(raw_unet.state_dict())
    ema_unet.requires_grad_(False)
    ema_unet.eval()
    ema_unet = ema_unet.cpu()

    def update_ema(ema_model, model, decay):
        with torch.no_grad():
            src = model._orig_mod if hasattr(model, '_orig_mod') else model
            for ema_p, p in zip(ema_model.parameters(), src.parameters()):
                ema_p.data.mul_(decay).add_(p.data.cpu(), alpha=1 - decay)

    # ========== Resume ==========
    start_step  = 0
    start_epoch = 0
    if args.resume_from is not None:
        start_step, start_epoch = load_checkpoint(
            args.resume_from, unet, optimizer, lr_scheduler, device
        )
        if args.reset_lr:
            remaining_steps = (train_epochs - start_epoch) * math.ceil(
                math.ceil(len(dataset) / batch_size) / gradient_accumulation_steps
            )
            remaining_warmup = remaining_steps // 20
            def warmup_cosine_lambda(s):
                if s < remaining_warmup:
                    return s / max(1, remaining_warmup)
                p = (s - remaining_warmup) / max(1, remaining_steps - remaining_warmup)
                return 0.5 * (1.0 + math.cos(math.pi * p))
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda)
            print(f"[reset_lr] New cosine schedule: {remaining_steps} steps, warmup {remaining_warmup}")

    print(f"[setup] total_steps={total_steps}, warmup_steps={warmup_steps}", flush=True)
    print("[train] Starting training loop...", flush=True)

    # ========== Training Loop ==========
    step = start_step
    loss_accumulated = 0.0
    optimizer.zero_grad()
    train_start_time = datetime.now()

    remaining_epochs = train_epochs - start_epoch
    pbar = tqdm(total=remaining_epochs * len(dataloader), desc="Training", initial=0)

    for epoch in range(start_epoch, train_epochs):
        print(f"[train] Epoch {epoch}/{train_epochs} started | elapsed={datetime.now()-train_start_time}", flush=True)
        for batch in dataloader:
            step += 1
            pixel_values = batch.to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = unet(noisy_latents, timesteps).sample
                v_target = noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(noise_pred, v_target, reduction="none").mean(dim=(1, 2, 3))
                snr = noise_scheduler.alphas_cumprod.to(device)[timesteps] / (
                    1 - noise_scheduler.alphas_cumprod.to(device)[timesteps]
                )
                min_snr_weights = torch.clamp(snr, max=5.0) / (snr + 1)
                loss = (loss * min_snr_weights).mean()

            with torch.no_grad():
                per_sample_loss = F.mse_loss(noise_pred.float(), v_target.float(), reduction="none").mean(dim=(1, 2, 3))
                early_mask = timesteps < 333
                mid_mask   = (timesteps >= 333) & (timesteps < 667)
                late_mask  = timesteps >= 667
                loss_early = per_sample_loss[early_mask].mean().item() if early_mask.any() else None
                loss_mid   = per_sample_loss[mid_mask].mean().item()   if mid_mask.any()   else None
                loss_late  = per_sample_loss[late_mask].mean().item()  if late_mask.any()  else None

            loss_scalar = loss.item()
            scaler.scale(loss / gradient_accumulation_steps).backward()
            loss_accumulated += loss_scalar

            if step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0).item()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                update_ema(ema_unet, unet, ema_decay)

                avg_loss = loss_accumulated / gradient_accumulation_steps
                current_lr = lr_scheduler.get_last_lr()[0]

                log_dict = {
                    "train/loss":      avg_loss,
                    "train/lr":        current_lr,
                    "train/grad_norm": grad_norm,
                    "train/epoch":     epoch,
                    "train/step":      step,
                }
                if loss_early is not None: log_dict["train/loss_early"] = loss_early
                if loss_mid   is not None: log_dict["train/loss_mid"]   = loss_mid
                if loss_late  is not None: log_dict["train/loss_late"]  = loss_late
                wandb.log(log_dict, step=step)
                loss_accumulated = 0.0
                if (step // gradient_accumulation_steps) % 100 == 0:
                    elapsed = datetime.now() - train_start_time
                    print(f"[train] step={step} epoch={epoch} loss={avg_loss:.4f} lr={current_lr:.2e} elapsed={elapsed}", flush=True)

            pbar.set_postfix({"loss": f"{loss_scalar:.4f}", "epoch": epoch})
            pbar.update(1)

            if step % eval_freq == 0:
                ema_unet = ema_unet.to(device)
                generate_and_save_images(unet, vae, noise_scheduler, step, device, sample_folder, ema_unet=ema_unet)
                ema_unet = ema_unet.cpu()
                torch.cuda.empty_cache()
                unet.train()

            if step % save_freq == 0:
                ema_unet = ema_unet.to(device)
                save_checkpoint(ckpt_folder, unet, ema_unet, optimizer, lr_scheduler, step, epoch)
                ema_unet = ema_unet.cpu()
                torch.cuda.empty_cache()

    ema_unet = ema_unet.to(device)
    save_checkpoint(ckpt_folder, unet, ema_unet, optimizer, lr_scheduler, "final", epoch)
    ema_unet = ema_unet.cpu()
    print("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    train()
