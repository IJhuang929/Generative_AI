import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import open_clip
from tqdm import tqdm

# ==========================================
# ★ 請修改這裡的路徑與參數
# ==========================================
GENERATE_CSV_PATH = "./data/generate.csv"   # generate.csv 放在 scripts/data/
CHECKPOINT_PATH   = "../model.pth"          # 模型權重在根目錄
OUTPUT_DIR        = "../generated_images"   # 圖片輸出到根目錄

BATCH_SIZE  = 32     # CUDA 建議 32~64，VRAM 不夠再往下調
CFG_SCALE   = 3.5
TIMESTEPS   = 1000   # 訓練用的總步數，不要動

# DDIM 加速：實際只跑這幾步，品質幾乎不變
# 200 步 → 約 5x 加速；100 步 → 約 10x 加速
DDIM_STEPS  = 200

# ==========================================
# 自動偵測裝置
# ==========================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"使用裝置: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Model 定義（與訓練時完全相同）
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
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.mlp   = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_c))
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_c),  nn.SiLU(), nn.Conv2d(in_c,  out_c, 3, padding=1))
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
        self.norm  = nn.GroupNorm(8, channels)
        self.qkv   = nn.Conv2d(channels, channels * 3, 1)
        self.proj  = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).view(B, 3, self.heads, C // self.heads, H * W).transpose(-1, -2)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(B, C, H, W))

class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim=512, heads=4):
        super().__init__()
        self.heads = heads
        self.norm  = nn.GroupNorm(8, channels)
        self.to_q  = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Linear(context_dim, channels * 2)
        self.proj  = nn.Conv2d(channels, channels, 1)
    def forward(self, x, context):
        B, C, H, W = x.shape
        seq_len = context.shape[1]
        h  = self.norm(x)
        q  = self.to_q(h).view(B, self.heads, C // self.heads, H * W).transpose(-1, -2)
        kv = self.to_kv(context).view(B, seq_len, 2, self.heads, C // self.heads)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(B, C, H, W))

class SimpleUNet(nn.Module):
    def __init__(self, base_channels=128):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.init_conv  = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down1_res  = ResBlock(base_channels,     base_channels * 2, time_dim)
        self.down1_pool = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.down2_res  = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down2_attn = CrossAttention(base_channels * 4, 512)
        self.down2_pool = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.down3_res  = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.down3_attn = CrossAttention(base_channels * 4, 512)
        self.down3_pool = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.mid_res1      = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_self_attn = SelfAttention(base_channels * 4)
        self.mid_res2      = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up1_up   = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.up1_res  = ResBlock(base_channels * 8, base_channels * 4, time_dim)
        self.up1_attn = CrossAttention(base_channels * 4, 512)
        self.up2_up   = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.up2_res  = ResBlock(base_channels * 6, base_channels * 2, time_dim)
        self.up2_attn = CrossAttention(base_channels * 2, 512)
        self.up3_up   = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.up3_res  = ResBlock(base_channels * 3, base_channels, time_dim)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels), nn.SiLU(), nn.Conv2d(base_channels, 3, 3, padding=1)
        )
    def forward(self, x, t, context):
        t_emb = self.time_mlp(t)
        x1    = self.init_conv(x)
        x2    = self.down1_res(x1, t_emb);  skip1 = x2;  x2 = self.down1_pool(x2)
        x3    = self.down2_res(x2, t_emb);  x3 = self.down2_attn(x3, context);  skip2 = x3;  x3 = self.down2_pool(x3)
        x4    = self.down3_res(x3, t_emb);  x4 = self.down3_attn(x4, context);  skip3 = x4;  x4 = self.down3_pool(x4)
        xm    = self.mid_res1(x4, t_emb);   xm = self.mid_self_attn(xm);        xm = self.mid_res2(xm, t_emb)
        xu1   = self.up1_up(xm);   xu1 = self.up1_res(torch.cat([xu1, skip3], 1), t_emb);  xu1 = self.up1_attn(xu1, context)
        xu2   = self.up2_up(xu1);  xu2 = self.up2_res(torch.cat([xu2, skip2], 1), t_emb);  xu2 = self.up2_attn(xu2, context)
        xu3   = self.up3_up(xu2);  xu3 = self.up3_res(torch.cat([xu3, skip1], 1), t_emb)
        return self.final_conv(xu3)

# ==========================================
# DDIM Sampler
# ==========================================
class DDIMSampler:
    """
    用 DDIM 跳步採樣，只需 DDIM_STEPS 步即可完成去噪。
    數學上等價於 DDPM，品質幾乎不變，速度快 5~10x。
    """
    def __init__(self, timesteps=1000, ddim_steps=200, device="cuda"):
        self.T        = timesteps
        self.S        = ddim_steps
        self.device   = device

        betas          = torch.linspace(1e-4, 0.02, timesteps).to(device)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod

        # 從 T 步中均勻挑選 S 個 timestep（倒序）
        step_ratio   = timesteps // ddim_steps
        self.ddim_ts = torch.arange(0, timesteps, step_ratio).flip(0).to(device)  # [S]

    @torch.no_grad()
    def sample(self, model, shape, cond, uncond, cfg_scale, eta=0.0):
        """
        eta=0.0 → 完全確定性（DDIM）
        eta=1.0 → 等同 DDPM（加回隨機性）
        """
        x = torch.randn(shape, device=self.device)
        B = shape[0]

        for i, t in enumerate(tqdm(self.ddim_ts, desc="DDIM Sampling", leave=False)):
            t_tensor = torch.full((B,), t.item(), device=self.device, dtype=torch.long)

            # CFG：有條件 / 無條件各推一次
            noise_uncond = model(x, t_tensor, uncond)
            noise_cond   = model(x, t_tensor, cond)
            noise_pred   = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # 取當前與前一步的 alpha_cumprod
            a_t  = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            t_prev = self.ddim_ts[i + 1] if i + 1 < len(self.ddim_ts) else torch.tensor(0, device=self.device)
            a_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)

            # 預測 x0
            sqrt_a_t   = a_t.sqrt()
            sqrt_1m_at = (1 - a_t).sqrt()
            x0_pred    = (x - sqrt_1m_at * noise_pred) / sqrt_a_t
            x0_pred    = x0_pred.clamp(-1, 1)

            # DDIM 更新公式
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).sqrt() * noise_pred
            noise  = sigma * torch.randn_like(x) if eta > 0 else 0
            x      = a_prev.sqrt() * x0_pred + dir_xt + noise

        return x

# ==========================================
# 載入模型
# ==========================================
print("載入 UNet 權重...")
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

sampler = DDIMSampler(timesteps=TIMESTEPS, ddim_steps=DDIM_STEPS, device=DEVICE)

# ==========================================
# 載入 CLIP，預先 encode 所有 prompt
# ==========================================
print("載入 CLIP 模型...")
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
tokenizer        = open_clip.get_tokenizer("ViT-B-32-quickgelu")
clip_model       = clip_model.to(DEVICE).eval()

def encode_prompt(prompt: str) -> torch.Tensor:
    with torch.no_grad():
        tokens = tokenizer([prompt]).to(DEVICE)
        x = clip_model.token_embedding(tokens)
        x = x + clip_model.positional_embedding
        x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
        x = clip_model.ln_final(x)
    return x.squeeze(0).cpu().float()

print("Encoding 空字串 context（CFG 用）...")
empty_context = encode_prompt("")

print("讀取 generate.csv 並 encode 所有 prompt...")
df = pd.read_csv(GENERATE_CSV_PATH)
all_contexts = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding prompts"):
    # prompt 欄有值就直接用，否則從 animal + object 組合
    if "prompt" in row and pd.notna(row["prompt"]) and str(row["prompt"]).strip():
        prompt = str(row["prompt"]).strip()
    else:
        prompt = f"a {row['animal']} and a {row['object']}"
    all_contexts.append(encode_prompt(prompt))

del clip_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ==========================================
# 批次生成（DDIM）
# ==========================================
print(f"\n開始生成 {len(df)} 張圖片")
print(f"  DDIM steps : {DDIM_STEPS}（原 DDPM 1000 步，加速 {TIMESTEPS // DDIM_STEPS}x）")
print(f"  CFG scale  : {CFG_SCALE}")
print(f"  Batch size : {BATCH_SIZE}")

with torch.no_grad():
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Generating batches"):
        batch_df  = df.iloc[i : i + BATCH_SIZE]
        B         = len(batch_df)
        batch_ctx = torch.stack(all_contexts[i : i + BATCH_SIZE]).to(DEVICE)
        uncond    = empty_context.unsqueeze(0).repeat(B, 1, 1).to(DEVICE)

        x = sampler.sample(
            model     = model,
            shape     = (B, 3, 64, 64),
            cond      = batch_ctx,
            uncond    = uncond,
            cfg_scale = CFG_SCALE,
        )

        x = ((x + 1) / 2).clamp(0, 1)

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            fname = str(row["id"])
            if not fname.endswith(".png"):
                fname += ".png"
            save_image(x[idx], os.path.join(OUTPUT_DIR, fname))

print(f"\n✅ 完成！圖片已存到 {OUTPUT_DIR}/")