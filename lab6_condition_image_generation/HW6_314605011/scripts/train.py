import os
import math
import random
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import open_clip
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import wandb

# ==========================================
# 1. 核心超參數
# ==========================================
BATCH_SIZE    = 64
EPOCHS        = 1000
LEARNING_RATE = 2e-4
TIMESTEPS     = 1000
CFG_DROP_RATE = 0.1
EMA_DECAY     = 0.9999

# ==========================================
# 路徑設定（由 argparse 覆蓋）
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--csv",      default="train_filtered.csv", help="訓練用 CSV 路徑")
parser.add_argument("--img_dir",  default="trainset",           help="訓練圖片資料夾")
parser.add_argument("--save_dir", default="checkpoints",        help="Checkpoint 儲存目錄")
args = parser.parse_args()

CSV_PATH = args.csv
IMG_DIR  = args.img_dir
SAVE_DIR = args.save_dir

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Dataset
# ==========================================
class BrainrotDataset(Dataset):
    def __init__(self, csv_path, img_dir, image_size=64, device="cuda", min_per_combo=30):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.device = device

        rows = []
        for (_, _), group in self.df.groupby(["animal", "object"]):
            shortage = max(0, min_per_combo - len(group))
            if shortage > 0:
                rows.append(group.sample(shortage, replace=True, random_state=42))
        if rows:
            self.df = pd.concat([self.df] + rows, ignore_index=True)
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.clip_cache = self._precompute_clip_embeddings()

    def _precompute_clip_embeddings(self):
        print("Precomputing CLIP embeddings (77 tokens)...")
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        model = model.to(self.device).eval()

        prompts = {f"a {row.animal} and a {row.object}" for row in self.df.itertuples()}
        prompts.add("")

        cache = {}
        with torch.no_grad():
            for prompt in prompts:
                tokens = tokenizer([prompt]).to(self.device)
                x = model.token_embedding(tokens)
                x = x + model.positional_embedding
                x = model.transformer(x, attn_mask=model.attn_mask)
                x = model.ln_final(x)
                cache[prompt] = x.squeeze(0).cpu().float()

        self.empty_context = cache[""].clone()
        del model
        torch.cuda.empty_cache()
        return cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = f"a {row['animal']} and a {row['object']}"
        tokens = self.clip_cache[prompt]

        if random.random() < CFG_DROP_RATE:
            tokens = self.empty_context.clone()

        img_path = os.path.join(self.img_dir, str(row["id"]))
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), tokens, prompt

# ==========================================
# 3. Diffusion 基礎工具與 EMA
# ==========================================
class GaussianDiffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ==========================================
# 4. 手刻 UNet 架構模組
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_c))
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_c), nn.SiLU(), nn.Conv2d(in_c, out_c, 3, padding=1))
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_c), nn.SiLU(), nn.Conv2d(out_c, out_c, 3, padding=1))
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.mlp(t_emb)[..., None, None]
        h = self.conv2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).view(B, 3, self.heads, C // self.heads, H * W).transpose(-1, -2)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(attn_out)

class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim=512, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x, context):
        B, C, H, W = x.shape
        seq_len = context.shape[1]
        h = self.norm(x)
        q = self.to_q(h).view(B, self.heads, C // self.heads, H * W).transpose(-1, -2)
        kv = self.to_kv(context).view(B, seq_len, 2, self.heads, C // self.heads)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(attn_out)

class SimpleUNet(nn.Module):
    def __init__(self, base_channels=128):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.init_conv = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down1_res = ResBlock(base_channels, base_channels * 2, time_dim)
        self.down1_pool = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.down2_res = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down2_attn = CrossAttention(base_channels * 4, 512)
        self.down2_pool = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.down3_res = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.down3_attn = CrossAttention(base_channels * 4, 512)
        self.down3_pool = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.mid_res1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_self_attn = SelfAttention(base_channels * 4)
        self.mid_res2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up1_up = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.up1_res = ResBlock(base_channels * 8, base_channels * 4, time_dim)
        self.up1_attn = CrossAttention(base_channels * 4, 512)
        self.up2_up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.up2_res = ResBlock(base_channels * 6, base_channels * 2, time_dim)
        self.up2_attn = CrossAttention(base_channels * 2, 512)
        self.up3_up = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.up3_res = ResBlock(base_channels * 3, base_channels, time_dim)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels), nn.SiLU(), nn.Conv2d(base_channels, 3, 3, padding=1)
        )
    def forward(self, x, t, context):
        t_emb = self.time_mlp(t)
        x1 = self.init_conv(x)
        x2 = self.down1_res(x1, t_emb)
        skip1 = x2
        x2 = self.down1_pool(x2)
        x3 = self.down2_res(x2, t_emb)
        x3 = self.down2_attn(x3, context)
        skip2 = x3
        x3 = self.down2_pool(x3)
        x4 = self.down3_res(x3, t_emb)
        x4 = self.down3_attn(x4, context)
        skip3 = x4
        x4 = self.down3_pool(x4)
        x_mid = self.mid_res1(x4, t_emb)
        x_mid = self.mid_self_attn(x_mid)
        x_mid = self.mid_res2(x_mid, t_emb)
        x_up1 = self.up1_up(x_mid)
        x_up1 = self.up1_res(torch.cat([x_up1, skip3], dim=1), t_emb)
        x_up1 = self.up1_attn(x_up1, context)
        x_up2 = self.up2_up(x_up1)
        x_up2 = self.up2_res(torch.cat([x_up2, skip2], dim=1), t_emb)
        x_up2 = self.up2_attn(x_up2, context)
        x_up3 = self.up3_up(x_up2)
        x_up3 = self.up3_res(torch.cat([x_up3, skip1], dim=1), t_emb)
        return self.final_conv(x_up3)

# ==========================================
# 5. DDPM 產圖函數 (給 WandB 監控用)
# ==========================================
@torch.no_grad()
def generate_samples_for_wandb(model, diffusion, dataset, device, cfg_scale=4.0):
    model.eval()
    sample_prompts = ["a dog and a ball", "a cat and a box", "a bird and a tree", "a fish and a boat"]

    cond_tokens = []
    for p in sample_prompts:
        if p in dataset.clip_cache:
            cond_tokens.append(dataset.clip_cache[p])
        else:
            cond_tokens.append(dataset.empty_context)

    cond_tokens = torch.stack(cond_tokens).to(device)
    uncond_tokens = dataset.empty_context.clone().unsqueeze(0).repeat(4, 1, 1).to(device)

    x = torch.randn(4, 3, 64, 64, device=device)

    for t in tqdm(reversed(range(0, TIMESTEPS)), desc="Sampling", leave=False):
        t_tensor = torch.full((4,), t, device=device, dtype=torch.long)

        noise_pred_uncond = model(x, t_tensor, uncond_tokens)
        noise_pred_cond = model(x, t_tensor, cond_tokens)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        betas_t = diffusion.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = diffusion.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        posterior_variance_t = diffusion.posterior_variance[t].view(-1, 1, 1, 1)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t == 0:
            x = model_mean
        else:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance_t) * noise

    x = (x + 1) / 2
    x = x.clamp(0, 1)

    images_to_log = []
    for i in range(4):
        images_to_log.append(wandb.Image(x[i], caption=sample_prompts[i]))
    return images_to_log

# ==========================================
# 6. 主訓練迴圈
# ==========================================
def train():
    print(f"Using device: {DEVICE}")

    wandb.init(
        project="brainrot-diffusion",
        name="unet-cfg-ema-warmup",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "timesteps": TIMESTEPS,
            "cfg_drop_rate": CFG_DROP_RATE,
            "ema_decay": EMA_DECAY,
            "warmup_ratio": 0.05
        }
    )

    dataset = BrainrotDataset(csv_path=CSV_PATH, img_dir=IMG_DIR, device=DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = SimpleUNet().to(DEVICE)
    diffusion = GaussianDiffusion(timesteps=TIMESTEPS, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    total_steps = len(dataloader) * EPOCHS
    warmup_steps = int(total_steps * 0.05)

    print(f"總步數: {total_steps}, Warmup 步數: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    ema = EMA(model, EMA_DECAY)

    global_step = 0

    print("開始訓練...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_epoch = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for images, tokens, _ in pbar:
            images = images.to(DEVICE)
            tokens = tokens.to(DEVICE)
            B = images.shape[0]

            t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()
            noise = torch.randn_like(images).to(DEVICE)

            x_noisy = diffusion.q_sample(images, t, noise)

            optimizer.zero_grad()

            with autocast():
                noise_pred = model(x_noisy, t, tokens)
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            ema.update()

            global_step += 1
            loss_epoch += loss.item()
            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

            wandb.log({
                "train/loss": loss.item(),
                "train/lr": current_lr,
                "train/step": global_step,
                "train/epoch": epoch
            })

        avg_loss = loss_epoch / len(dataloader)
        print(f"Epoch {epoch}/{EPOCHS} - Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6e}")

        if epoch % 50 == 0:
            ema.apply_shadow()

            save_path = os.path.join(SAVE_DIR, f"unet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

            print("正在生成測試圖片傳送到 W&B...")
            generated_images = generate_samples_for_wandb(model, diffusion, dataset, DEVICE, cfg_scale=4.0)
            wandb.log({"eval/generated_samples": generated_images, "train/epoch": epoch})

            ema.restore()

    wandb.finish()


train()
