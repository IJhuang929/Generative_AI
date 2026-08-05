import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────
# 1. Time Embedding
# ─────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """
    把整數 timestep t 轉成向量。
    步驟：sinusoidal encoding → Linear → SiLU → Linear
    """
    def __init__(self, base_dim, time_dim):
        # base_dim: sinusoidal 維度 (= model 最小 ch，這裡是 128)
        # time_dim: 輸出維度 (= 1024)
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def _sinusoidal(self, t):
        # t: (B,) 整數 tensor
        # 輸出: (B, base_dim)
        half = self.base_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                           # (half,)
        args = t[:, None].float() * freqs[None, :] # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, base_dim)
        return emb

    def forward(self, t):
        emb = self._sinusoidal(t)   # (B, base_dim)
        emb = self.mlp(emb)         # (B, time_dim)
        return emb


# ─────────────────────────────────────────────
# 2. ResBlock
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    主路徑：Conv → GroupNorm → SiLU → Conv
    timestep 注入：AdaGN（scale + shift）
    shortcut：1×1 Conv（當 in_ch ≠ out_ch 時）
    """
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1, num_groups=32):
        super().__init__()

        # 主路徑 第一段
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # timestep projection → 產生 scale 和 shift，各 out_ch 維
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)

        # 主路徑 第二段
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # shortcut
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        # x:        (B, in_ch, H, W)
        # time_emb: (B, time_dim)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)                           # (B, out_ch, H, W)

        # AdaGN：把 time_emb 投影成 scale, shift
        t = self.time_proj(time_emb)                # (B, out_ch*2)
        scale, shift = t.chunk(2, dim=-1)           # 各 (B, out_ch)
        scale = scale[:, :, None, None]             # (B, out_ch, 1, 1)
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift     # AdaGN
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)                           # (B, out_ch, H, W)

        return h + self.shortcut(x)                 # residual


# ─────────────────────────────────────────────
# 3. CrossAttention
# ─────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    讓 feature map 的每個位置查詢 condition embedding。

    Q: feature map  (B, H*W, ch)
    K: condition    (B, 77,  cond_dim)
    V: condition    (B, 77,  cond_dim)

    流程：
      feature map → reshape → Q
      condition   → K, V
      Attention(Q, K, V) → reshape 回原本空間
      + residual
    """
    def __init__(self, ch, cond_dim=512, num_heads=8, num_groups=32):
        super().__init__()
        self.ch = ch
        self.num_heads = num_heads

        # attention 前的 norm
        self.norm = nn.GroupNorm(num_groups, ch)

        # Q 從 feature map 來，K/V 從 condition 來
        # 統一投影到 ch 維，方便 attention 計算
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(cond_dim, ch)
        self.to_v = nn.Linear(cond_dim, ch)

        # attention 輸出投影
        self.to_out = nn.Linear(ch, ch)

    def forward(self, x, cond):
        # x:    (B, ch, H, W)
        # cond: (B, 77, 512)
        B, C, H, W = x.shape

        # norm + reshape feature map 成序列
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)    # (B, H*W, C)

        # Q, K, V
        Q = self.to_q(h)                                 # (B, H*W, ch)
        K = self.to_k(cond)                              # (B, 77,  ch)
        V = self.to_v(cond)                              # (B, 77,  ch)

        # 拆成 multi-head
        head_dim = self.ch // self.num_heads
        def split_heads(t):
            # (B, seq, ch) → (B, heads, seq, head_dim)
            bs, seq, _ = t.shape
            t = t.reshape(bs, seq, self.num_heads, head_dim)
            return t.permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # scaled dot-product attention
        # PyTorch 2.0+ 有 F.scaled_dot_product_attention，效率更好
        attn_out = F.scaled_dot_product_attention(Q, K, V)  # (B, heads, H*W, head_dim)

        # 合回單頭
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, H * W, C)  # (B, H*W, ch)
        attn_out = self.to_out(attn_out)                               # (B, H*W, ch)

        # reshape 回空間 + residual
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)      # (B, ch, H, W)
        return x + attn_out


# ─────────────────────────────────────────────
# 4. SelfAttention
# ─────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    讓 feature map 內部各位置互相 attend，學習空間關係。
    Q / K / V 全來自 feature map 本身。
    """
    def __init__(self, ch, num_heads=8, num_groups=32, max_size=64):
        super().__init__()
        self.ch = ch
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, ch)
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(ch, ch)
        self.to_v = nn.Linear(ch, ch)
        self.to_out = nn.Linear(ch, ch)
        self.pos_emb = nn.Embedding(max_size * max_size, ch)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)

        seq_len = H * W
        pos_ids = torch.arange(seq_len, device=x.device)
        h = h + self.pos_emb(pos_ids).unsqueeze(0)    # (B, H*W, C)

        Q = self.to_q(h)
        K = self.to_k(h)
        V = self.to_v(h)

        head_dim = self.ch // self.num_heads
        def split_heads(t):
            bs, seq, _ = t.shape
            return t.reshape(bs, seq, self.num_heads, head_dim).permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        attn_out = self.to_out(attn_out)
        return x + attn_out.permute(0, 2, 1).reshape(B, C, H, W)


# ─────────────────────────────────────────────
# 5. UNet
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Conditional UNet for pixel-space DDPM.

    Default: (B, 3, 64, 64) pixel images
    channels:  (96, 192, 384, 384)  對應解析度 (64, 32, 16, 8)
    每個 level: ResBlock × 3
    SelfAttn:   level {1,2,3}（32/16/8×8）；CrossAttn：level {2,3} + middle（16/8×8）
    """
    def __init__(
        self,
        in_ch             = 3,
        base_ch           = 96,
        ch_mult           = (1, 2, 4, 4),   # → (96, 192, 384, 384)
        num_res           = 3,               # ResBlock per level
        cond_dim          = 512,
        time_dim          = 576,
        num_heads         = 8,
        dropout           = 0.1,
        num_groups        = 32,
        attn_levels       = None,            # SelfAttn levels; None → {1,2,3}
        cross_attn_levels = None,            # CrossAttn levels; None → {2,3}
    ):
        super().__init__()
        channels = [base_ch * m for m in ch_mult]
        if attn_levels is None:
            n = len(ch_mult)
            attn_levels = set(range(1, n))   # {1, 2, 3} - skip level 0 (64×64)
        attn_levels = set(attn_levels)

        if cross_attn_levels is None:
            n = len(ch_mult)
            cross_attn_levels = {n - 2, n - 1}   # {2, 3}
        cross_attn_levels = set(cross_attn_levels)

        # ── time embedding ──
        self.time_emb = TimeEmbedding(base_ch, time_dim)

        # ── condition projection（cond → time_dim，與 time_emb 相加）──
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── input conv ──
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # ── encoder ──
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_c = base_ch
        self.enc_ch = []   # 記錄每層輸出 ch，給 skip connection 用

        for i, out_c in enumerate(channels):
            level_blocks = nn.ModuleList()
            for j in range(num_res):
                level_blocks.append(
                    ResBlock(in_c, out_c, time_dim, dropout, num_groups)
                )
                in_c = out_c
            if i in attn_levels:
                level_blocks.append(SelfAttention(out_c, num_heads, num_groups))
            if i in cross_attn_levels:
                level_blocks.append(CrossAttention(out_c, cond_dim, num_heads, num_groups))
            self.enc_blocks.append(level_blocks)
            self.enc_ch.append(out_c)

            # 最後一層不 downsample（進 middle）
            if i < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1)
                )
            else:
                self.downsamples.append(None)

        # ── middle ──
        mid_ch = channels[-1]
        self.mid_block1     = ResBlock(mid_ch, mid_ch, time_dim, dropout, num_groups)
        self.mid_self_attn  = SelfAttention(mid_ch, num_heads, num_groups)
        self.mid_attn       = CrossAttention(mid_ch, cond_dim, num_heads, num_groups)
        self.mid_block2     = ResBlock(mid_ch, mid_ch, time_dim, dropout, num_groups)

        # ── decoder ──
        self.dec_blocks  = nn.ModuleList()
        self.upsamples   = nn.ModuleList()

        for i, out_c in enumerate(reversed(channels)):
            skip_c = self.enc_ch[-(i + 1)]   # 對應 encoder 層的輸出 ch
            level_blocks = nn.ModuleList()

            for j in range(num_res):
                # 第一個 ResBlock 接收 upsample + skip concat
                block_in = in_c + skip_c if j == 0 else out_c
                level_blocks.append(
                    ResBlock(block_in, out_c, time_dim, dropout, num_groups)
                )
                in_c = out_c

            attn_idx = len(channels) - 1 - i   # 對應 encoder 的 level index
            if attn_idx in attn_levels:
                level_blocks.append(SelfAttention(out_c, num_heads, num_groups))
            if attn_idx in cross_attn_levels:
                level_blocks.append(CrossAttention(out_c, cond_dim, num_heads, num_groups))
            self.dec_blocks.append(level_blocks)

            # 最後一層（最淺，64×64）不需要 upsample
            if i < len(channels) - 1:
                self.upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    )
                )
            else:
                self.upsamples.append(None)

        # ── output ──
        self.out_norm = nn.GroupNorm(num_groups, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)

        self.use_checkpoint = False

    def enable_gradient_checkpointing(self):
        self.use_checkpoint = True
        print("Gradient checkpointing enabled (saves ~40% activation memory, ~20% slower)")

    def forward(self, x, t, cond, eos_idx=None):
        # x:       (B, 3, 64, 64)
        # t:       (B,)
        # cond:    (B, 77, 512)  CLIP token-level embedding
        # eos_idx: (B,) int — EOS token position for AdaGN global feature

        if eos_idx is not None:
            batch_idx = torch.arange(cond.shape[0], device=cond.device)
            cond_global = cond[batch_idx, eos_idx]            # (B, 512)
        else:
            cond_global = cond[:, 0]

        time_emb = self.time_emb(t) + self.cond_proj(cond_global)    # (B, time_dim)

        # input conv
        h = self.in_conv(x)

        # encoder — 存 skip connections
        skips = []
        for i, (level_blocks, ds) in enumerate(zip(self.enc_blocks, self.downsamples)):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    if self.use_checkpoint and self.training:
                        h = checkpoint(block, h, time_emb, use_reentrant=False)
                    else:
                        h = block(h, time_emb)
                elif isinstance(block, SelfAttention):
                    if self.use_checkpoint and self.training:
                        h = checkpoint(block, h, use_reentrant=False)
                    else:
                        h = block(h)
                else:
                    if self.use_checkpoint and self.training:
                        h = checkpoint(block, h, cond, use_reentrant=False)
                    else:
                        h = block(h, cond)
            skips.append(h)
            if ds is not None:
                h = ds(h)

        # middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_self_attn(h)
        h = self.mid_attn(h, cond)
        h = self.mid_block2(h, time_emb)

        # decoder
        for i, (level_blocks, us) in enumerate(zip(self.dec_blocks, self.upsamples)):
            skip = skips[-(i + 1)]
            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                if isinstance(block, ResBlock):
                    if self.use_checkpoint and self.training:
                        h = checkpoint(block, h, time_emb, use_reentrant=False)
                    else:
                        h = block(h, time_emb)
                elif isinstance(block, SelfAttention):
                    if self.use_checkpoint and self.training:
                        h = checkpoint(block, h, use_reentrant=False)
                    else:
                        h = block(h)
                else:
                    if self.use_checkpoint and self.training:
                        h = checkpoint(block, h, cond, use_reentrant=False)
                    else:
                        h = block(h, cond)

            if us is not None:
                h = us(h)

        # output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)              # (B, 3, 64, 64)
        return h


# ─────────────────────────────────────────────
# 快速測試
# ─────────────────────────────────────────────

if __name__ == "__main__":
    B = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TimeEmbedding
    te = TimeEmbedding(base_dim=128, time_dim=1024).to(device)
    t = torch.randint(0, 1000, (B,)).to(device) 
    time_emb = te(t)
    print(f"TimeEmbedding output: {time_emb.shape}")  # (4, 1024)

    # ResBlock - channel 改變
    rb1 = ResBlock(in_ch=128, out_ch=256, time_dim=1024).to(device)
    x = torch.randn(B, 128, 32, 32).to(device)
    out = rb1(x, time_emb)
    print(f"ResBlock (128→256):   {out.shape}")       # (4, 256, 32, 32)

    # ResBlock - channel 不變
    rb2 = ResBlock(in_ch=256, out_ch=256, time_dim=1024).to(device)
    out2 = rb2(out, time_emb)
    print(f"ResBlock (256→256):   {out2.shape}")      # (4, 256, 32, 32)

    # CrossAttention
    ca = CrossAttention(ch=512, cond_dim=512, num_heads=8).to(device)
    x_attn = torch.randn(B, 512, 16, 16).to(device)
    cond = torch.randn(B, 77, 512).to(device)
    out_attn = ca(x_attn, cond)
    print(f"CrossAttention:       {out_attn.shape}")  # (4, 512, 16, 16)

    # UNet
    model = UNet().to(device)
    x       = torch.randn(B, 3, 64, 64).to(device)
    t       = torch.randint(0, 1000, (B,)).to(device)
    cond    = torch.randn(B, 77, 512).to(device)
    eos_idx = torch.randint(5, 10, (B,)).to(device)
    out   = model(x, t, cond, eos_idx)
    print(f"UNet output:          {out.shape}")       # (4, 3, 64, 64)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:     {total/1e6:.2f}M")
    
