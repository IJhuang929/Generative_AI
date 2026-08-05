import os
import argparse
import torch
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, AutoencoderKL
from tqdm import tqdm


def build_scheduler(sampler: str, num_steps: int):
    """
    Build a numerically stable sampling scheduler.
    clip_sample=True + timestep_spacing="leading" prevents terminal-step explosion
    that causes all-black / RGB-noise outputs.
    """
    kwargs = dict(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=4.0,
        timestep_spacing="leading",
    )
    if sampler == "ddpm":
        scheduler = DDPMScheduler(**kwargs)
    elif sampler == "ddim":
        scheduler = DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"unknown sampler '{sampler}' (use 'ddpm' or 'ddim')")
    scheduler.set_timesteps(num_steps)
    return scheduler


@torch.no_grad()
def generate_and_save_images(unet, vae, scheduler, sampler, device,
                             save_folder, num_samples, batch_size=8, eta=0.0):
    unet.eval()
    vae.eval()

    os.makedirs(save_folder, exist_ok=True)
    to_pil = transforms.ToPILImage()
    sf = vae.config.scaling_factor

    total_batches = (num_samples + batch_size - 1) // batch_size
    sample_idx = 0
    n_black = n_white = 0

    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches), desc="Generating"):
            current_batch = min(batch_size, num_samples - sample_idx)

            latents = torch.randn(
                (current_batch, 4, 32, 32), device=device, dtype=torch.float32
            )

            for t in scheduler.timesteps:
                noise_pred = unet(latents, t).sample.float()
                step_kwargs = {"eta": eta} if sampler == "ddim" else {}
                latents = scheduler.step(
                    noise_pred, t, latents, **step_kwargs
                ).prev_sample.float()

            # Diagnostic on first batch
            if batch_idx == 0:
                lat_std = latents.std().item()
                flag = "OK" if 0.5 <= lat_std <= 2.0 else "!! ABNORMAL !!"
                print(f"[diag] final latent std={lat_std:.3f} (healthy 0.5-2.0) {flag}")

            images = vae.decode(latents / sf, return_dict=False)[0]
            images = (images.clamp(-1, 1) + 1) / 2

            # Saturation monitor
            m = images.mean(dim=(1, 2, 3))
            n_black += int((m < 0.02).sum())
            n_white += int((m > 0.98).sum())

            images = images.cpu()
            torch.cuda.empty_cache()

            for i in range(current_batch):
                to_pil(images[i]).save(os.path.join(save_folder, f"{sample_idx:04d}.png"))
                sample_idx += 1

    print(f"Saved {sample_idx} images to {save_folder}")
    print(f"[diag] saturated: black={n_black} white={n_white} "
          f"({100*(n_black+n_white)/max(sample_idx,1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="3070 Ti 8GB 建議用 8~16")
    parser.add_argument("--sampler", type=str, default="ddim",
                        choices=["ddpm", "ddim"])
    parser.add_argument("--num_steps", type=int, default=100,
                        help="DDIM: 50~200 steps; DDPM: 250~1000 steps")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity. Keep 0.0 for best FID.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading models on {device}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    unet = UNet2DModel.from_pretrained(args.checkpoint_dir).to(device)
    unet.requires_grad_(False)

    scheduler = build_scheduler(args.sampler, args.num_steps)
    print(f"Sampler={args.sampler}  steps={args.num_steps}  eta={args.eta}")

    generate_and_save_images(
        unet=unet, vae=vae, scheduler=scheduler, sampler=args.sampler,
        device=device, save_folder=args.output_dir,
        num_samples=args.num_samples, batch_size=args.batch_size, eta=args.eta,
    )