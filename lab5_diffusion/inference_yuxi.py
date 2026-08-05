import os
import argparse
import torch
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, AutoencoderKL
from tqdm import tqdm


def build_scheduler(sampler: str, num_steps: int):
    """Build a sampling scheduler that does NOT explode at the terminal step.

    Root-cause of the original all-black / RGB-noise outputs:
      The UNet is an *epsilon* model trained on the cosine (squaredcos_cap_v2)
      schedule, whose terminal step t=999 has alpha_cumprod ~ 1e-9.  With
      `timestep_spacing="trailing"` the FIRST reverse step lands exactly on
      t=999, so pred_x0 = (x_t - sqrt(1-a)*eps) / sqrt(a) divides a tiny eps
      residual by sqrt(a) ~ 3e-5 and blows up to std ~ 200 (real latents have
      std ~ 1).  With `clip_sample=False` nothing clamps it, the overscaled
      latent rides the whole deterministic trajectory, and the VAE decodes
      garbage.  eta=1.0 turned that garbage into pure black / white.

    Two independent fixes, both applied here for robustness:
      * clip_sample=True, clip_sample_range=4.0  -> clamp pred_x0 to +-4 sigma
        every step (real latents are ~N(0,1), so +-4 keeps 99.99% of signal
        while capping the explosion).
      * timestep_spacing="leading" -> never start exactly at t=999.
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
def generate_and_save_images(unet, vae, scheduler, sampler, device, save_folder,
                             num_samples, batch_size=32, eta=0.0):
    unet.eval()
    vae.eval()

    os.makedirs(save_folder, exist_ok=True)
    to_pil = transforms.ToPILImage()
    sf = vae.config.scaling_factor

    total_batches = (num_samples + batch_size - 1) // batch_size
    sample_idx = 0
    n_black = n_white = 0

    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches)):
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

            if batch_idx == 0:
                lat_std = latents.std().item()
                flag = "OK" if 0.5 <= lat_std <= 2.0 else "!! ABNORMAL !!"
                print(f"[diag] final latent std={lat_std:.3f} (healthy 0.5-2.0) {flag}")

            images = vae.decode(latents / sf, return_dict=False)[0]
            images = (images.clamp(-1, 1) + 1) / 2

            # cheap saturation monitor so a regression never silently ships
            m = images.mean(dim=(1, 2, 3))
            n_black += int((m < 0.02).sum())
            n_white += int((m > 0.98).sum())

            for i in range(current_batch):
                output_path = os.path.join(save_folder, f"{sample_idx:04d}.png")
                to_pil(images[i].cpu()).save(output_path)
                sample_idx += 1

    print(f"Saved {sample_idx} samples to {save_folder}")
    print(f"[diag] saturated outputs: black={n_black} white={n_white} "
          f"({100*(n_black+n_white)/max(sample_idx,1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained diffusion UNet model.")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/checkpoints/unet_final",
                        help="Path to the trained UNet checkpoint folder saved with save_pretrained().")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory where generated images will be saved.")
    parser.add_argument("--num_samples", type=int, default=3000,
                        help="Total number of images to generate.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for latent generation.")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                        help="ddpm (best FID, ~250 steps) or ddim (faster, ~50-100 steps).")
    parser.add_argument("--num_steps", type=int, default=250,
                        help="Number of denoising steps (250 for ddpm, 50-100 for ddim).")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity. Keep 0.0 (deterministic); >0 hurts FID here.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading VAE and UNet on {device}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    unet = UNet2DModel.from_pretrained(args.checkpoint_dir).to(device)
    unet.requires_grad_(False)

    scheduler = build_scheduler(args.sampler, args.num_steps)
    print(f"Sampler={args.sampler}  steps={args.num_steps}  eta={args.eta}  "
          f"spacing=leading  clip_sample=True(range=4.0)")

    generate_and_save_images(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        sampler=args.sampler,
        device=device,
        save_folder=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        eta=args.eta,
    )
