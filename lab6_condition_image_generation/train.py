import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import copy
import wandb
from datetime import datetime
import math
import argparse
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
from dataset import BrainrotDataset
from model import UNet
from diffusion import DDPMScheduler


ACCUM_STEPS = 4   # bs=16 * accum=4 = 有效 bs=64


# ─────────────────────────────────────────────
# Learning rate schedule：warmup + cosine decay
# ─────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# EMA update (ema fp16, model fp32)
# ─────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model, model, step, decay=0.999):
    # EMA Warmup：讓早期的 random init 快速被覆蓋
    current_decay = min(decay, (1 + step) / (10 + step))
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        # fp32 中間計算避免 fp16 denormal 累積誤差，再 cast 回 fp16 存
        updated = ema_p.data.float() * current_decay + p.data.float() * (1 - current_decay)
        ema_p.data.copy_(updated.half())


# ─────────────────────────────────────────────
# 單一 training step
# ─────────────────────────────────────────────

def train_step(model, scheduler, scaler, x0, cond, eos_idx, device, p_uncond=0.10):
    x0      = x0.to(device, non_blocking=True)
    cond    = cond.to(device, non_blocking=True).clone()       # (B, 77, 512)
    eos_idx = eos_idx.to(device, non_blocking=True)

    mask = torch.rand(cond.shape[0], device=device) < p_uncond
    cond[mask] = 0.0

    t = torch.randint(0, scheduler.T, (x0.shape[0],), device=device)
    xt, noise = scheduler.q_sample(x0, t)
    v_target = scheduler.get_v_target(x0, noise, t)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        v_pred = model(xt, t, cond, eos_idx)
        snr = scheduler.snr[t]
        weight = (snr.clamp(max=5.0) / (snr + 1)).to(device)
        loss_per_sample = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=(1, 2, 3))
        loss = (loss_per_sample * weight).mean()

    # timestep-grouped loss for monitoring
    with torch.no_grad():
        em = t < (scheduler.T // 3)
        mm = (t >= scheduler.T // 3) & (t < 2 * scheduler.T // 3)
        lm = t >= 2 * scheduler.T // 3
        le   = loss_per_sample[em].mean().item()  if em.any()  else float('nan')
        lmid = loss_per_sample[mm].mean().item()  if mm.any()  else float('nan')
        ll   = loss_per_sample[lm].mean().item()  if lm.any()  else float('nan')

    scaler.scale(loss / ACCUM_STEPS).backward()
    return loss.item(), t.float().mean().item(), le, lmid, ll


# ─────────────────────────────────────────────
# Visualize：生成 5 張圖存成 PNG
# ─────────────────────────────────────────────

VIZ_PROMPTS = [
    "a shark and a chair",
    "a cat and a cactus",
    "a dog and a airplane",
    "a frog and a toilet",
    "a bird and a drum",
]

@torch.no_grad()
def visualize(model, ema_model, scheduler, clip_cache, step, out_dir, device, cfg_scale=5.0):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    tokens_list = [clip_cache[p]["tokens"]  for p in VIZ_PROMPTS]
    eos_list    = [clip_cache[p]["eos_idx"] for p in VIZ_PROMPTS]
    cond    = torch.stack(tokens_list).to(device).half()                    # (5, 77, 512)
    eos_idx = torch.tensor(eos_list, device=device, dtype=torch.long)       # (5,)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        images = scheduler.ddim_sample(
            model           = ema_model,
            shape           = (5, 3, 64, 64),
            cond            = cond,
            eos_idx         = eos_idx,
            device          = device,
            ddim_steps      = 50,
            eta             = 0.0,
            cfg_scale       = cfg_scale,
            prediction_type = "v",
        )
    images = images.float()

    lat_std = images.std().item()
    if not (0.3 <= lat_std <= 1.5):
        tqdm.write(f"[viz][WARN] final std={lat_std:.2f} (expected ~0.5-1.0 for pixel [-1,1])")

    images = (images.clamp(-1, 1) + 1) / 2
    grid = torchvision.utils.make_grid(images, nrow=5, padding=2)
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    torchvision.utils.save_image(grid, path)
    tqdm.write(f"  → viz saved {path}")

    wandb_images = [
        wandb.Image(img.permute(1,2,0).cpu().numpy(), caption=prompt)
        for img, prompt in zip(images, VIZ_PROMPTS)
    ]

    torch.cuda.empty_cache()
    model.train()
    return wandb_images


# ─────────────────────────────────────────────
# Overfit test（sanity check）
# ─────────────────────────────────────────────

def overfit_test(num_steps=300, log_every=50):
    """Single-batch overfit sanity check. Loss < 0.1 within 300 steps = healthy."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BrainrotDataset(
        csv_path="train_filtered.csv", img_dir="trainset",
        image_size=64, device=device, latent_mode=False, min_per_combo=30,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    fixed_x0, fixed_cond, fixed_eos = next(iter(loader))
    fixed_x0   = fixed_x0.to(device)
    fixed_cond = fixed_cond.to(device)
    fixed_eos  = fixed_eos.to(device)

    model = UNet(
        in_ch=3, base_ch=96, ch_mult=(1, 2, 4, 4), num_res=3,
        time_dim=576, dropout=0.0,
    ).to(device)
    scheduler_ = DDPMScheduler(T=1000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    print(f"Overfit test: {num_steps} steps on {fixed_x0.shape[0]} images")
    model.train()
    recent = []

    for s in range(1, num_steps + 1):
        t = torch.randint(0, 1000, (fixed_x0.shape[0],), device=device)
        xt, noise = scheduler_.q_sample(fixed_x0, t)
        v_target = scheduler_.get_v_target(fixed_x0, noise, t)
        v_pred = model(xt, t, fixed_cond, fixed_eos)
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        recent.append(loss.item())
        if len(recent) > 50:
            recent.pop(0)

        if s % log_every == 0 or s == 1:
            avg = sum(recent) / len(recent)
            print(f"  step {s:4d} | loss {loss.item():.5f}  (avg-50: {avg:.5f})")

    final  = sum(recent) / len(recent)
    status = "PASS" if final < 0.1 else "FAIL"
    print(f"\n[{status}] avg last-50 loss = {final:.5f}  (expect < 0.1)")


# ─────────────────────────────────────────────
# 主訓練 loop
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    run_dir = os.path.join(args.ckpt_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    config = {
        "base_ch":          96,
        "ch_mult":          (1, 2, 4, 4),
        "num_res":          3,
        "time_dim":         576,
        "dropout":          0.10,
        "max_steps":        args.max_steps,
        "batch_size":       args.batch_size,
        "accum_steps":      ACCUM_STEPS,
        "effective_bs":     args.batch_size * ACCUM_STEPS,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "T":                1000,
        "warmup_pct":       0.05,
        "p_uncond":         0.10,
        "cfg_scale":        args.cfg_scale,
        "prediction_type":  "v",
        "loss_weighting":   "min_snr_gamma5_vpred",
        "cond_type":        "token_77_with_eos_adagn",
        "augment":          "hflip + light_jitter_0.02",
        "grad_checkpoint":  args.grad_checkpoint,
    }

    wandb.init(
        project = "conditional-ddpm",
        name    = datetime.now().strftime("%Y%m%d_%H%M%S"),
        config  = config,
        dir     = run_dir,
    )

    # dataset & dataloader
    dataset = BrainrotDataset(
        csv_path     = args.csv,
        img_dir      = args.img_dir,
        image_size   = 64,
        device       = device,
        latent_mode  = False,
        min_per_combo= 30,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # model & scheduler
    model = UNet(
        in_ch    = 3,
        base_ch  = 96,
        ch_mult  = (1, 2, 4, 4),
        num_res  = 3,
        time_dim = 576,
        dropout  = 0.10,
    ).to(device)
    scheduler = DDPMScheduler(T=1000).to(device)

    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # EMA model fp16，省 ~200MB
    ema_model = copy.deepcopy(model).half()
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    # optimizer & lr schedule
    MAX_STEPS        = args.max_steps
    optimizer_steps  = MAX_STEPS // ACCUM_STEPS
    optimizer        = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps     = int(optimizer_steps * 0.05)
    lr_scheduler     = get_lr_scheduler(optimizer, warmup_steps, optimizer_steps)
    scaler           = torch.cuda.amp.GradScaler()

    print(f"Max steps:    {MAX_STEPS}")
    print(f"Warmup steps: {warmup_steps} (optimizer steps)")
    print()

    # resume from checkpoint
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        # EMA 從 model 重新初始化，丟棄可能壞掉的舊 EMA weights
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.copy_(p.data.half())
        start_step = ckpt.get("step", ckpt.get("epoch", 0) * len(loader))
        print(f"Resumed from step {start_step}")

    viz_dir = os.path.join(run_dir, "viz")

    # training loop
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    ema_step = 0  # resets each run so warmup always applies, even on resume
    pbar = tqdm(total=MAX_STEPS, initial=step, desc="Training", unit="step")

    torch.cuda.empty_cache()
    print(f"Initial VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    while step < MAX_STEPS:
        for x0, cond, eos_idx in loader:
            if step >= MAX_STEPS:
                break

            model.train()
            try:
                loss_val, t_mean, le, lmid, ll = train_step(
                    model, scheduler, scaler, x0, cond, eos_idx, device
                )
            except torch.cuda.OutOfMemoryError:
                tqdm.write(f"OOM at step {step}, clearing cache and skipping batch")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step = (step + 1) // ACCUM_STEPS
                ema_step += 1
                update_ema(ema_model, model, step=ema_step)
                lr_scheduler.step()
            else:
                grad_norm = 0.0

            step += 1
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")
            wandb.log({
                "train/loss":       loss_val,
                "train/lr":         current_lr,
                "train/grad_norm":  grad_norm,
                "train/t_mean":     t_mean,
                "train/loss_early": le,
                "train/loss_mid":   lmid,
                "train/loss_late":  ll,
            }, step=step)

            if step % 1000 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved  = torch.cuda.memory_reserved() / 1e9
                tqdm.write(f"Step {step}/{MAX_STEPS} | loss={loss_val:.4f} | VRAM alloc={allocated:.2f}GB reserved={reserved:.2f}GB")
                torch.cuda.empty_cache()

            if step % args.save_every == 0 or step == MAX_STEPS:
                ckpt_path = os.path.join(run_dir, f"ckpt_{step:06d}.pth")
                torch.save({
                    "step":         step,
                    "model":        model.state_dict(),
                    "ema_model":    ema_model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }, ckpt_path)
                tqdm.write(f"  → saved {ckpt_path}")
                wandb_imgs = visualize(model, ema_model, scheduler, dataset.clip_cache,
                                       step, viz_dir, device, cfg_scale=args.cfg_scale)
                wandb.log({"viz/generated": wandb_imgs}, step=step)

    pbar.close()
    wandb.finish()
    print("Training complete.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit",         action="store_true")
    parser.add_argument("--overfit_steps",   type=int,   default=300)
    parser.add_argument("--csv",             type=str,   default="train_filtered.csv")
    parser.add_argument("--img_dir",         type=str,   default="trainset")
    parser.add_argument("--ckpt_dir",        type=str,   default="checkpoints")
    parser.add_argument("--resume",          type=str,   default=None)
    parser.add_argument("--max_steps",       type=int,   default=80_000)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--num_workers",     type=int,   default=2)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-2)
    parser.add_argument("--save_every",      type=int,   default=5000)
    parser.add_argument("--cfg_scale",       type=float, default=5.0)
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="OOM 時加上這個 flag 啟用 gradient checkpointing")
    args = parser.parse_args()

    if args.overfit:
        overfit_test(num_steps=args.overfit_steps)
    else:
        train(args)
