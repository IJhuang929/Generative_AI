import torch
import numpy as np


class DDPMScheduler:
    """
    DDPM scheduler with cosine noise schedule.

    Forward process（訓練）:
        xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε

    Reverse process（inference）:
        xₜ₋₁ = (1/√αₜ) · (xₜ - (1-αₜ)/√(1-ᾱₜ) · ε̂) + σₜ · z
    """

    def __init__(self, T=1000, s=0.008):
        # T: 總 timestep 數
        # s: cosine schedule 的 offset，避免 t=0 時 β 太小
        self.T = T

        # ── cosine schedule ──
        # 計算每個 timestep 的 ᾱₜ
        steps = np.arange(T + 1, dtype=np.float64)
        alphas_bar = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]          # normalize，讓 ᾱ₀ = 1

        # βₜ 從 ᾱₜ 反推，並 cap 在 0.999
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        betas = np.clip(betas, 0, 0.999)                 # (T,)

        # 從 βₜ 推導其他係數
        alphas     = 1 - betas                           # αₜ
        alphas_bar = np.cumprod(alphas)                  # ᾱₜ = ∏αᵢ

        # 存成 tensor，register 為 buffer（不是參數，但會隨 device 移動）
        self.register("betas",             betas)
        self.register("alphas",            alphas)
        self.register("alphas_bar",        alphas_bar)
        self.register("sqrt_alphas_bar",   np.sqrt(alphas_bar))
        self.register("sqrt_one_minus_ab", np.sqrt(1 - alphas_bar))

        # Min-SNR weighting 用的 SNR(t) = ᾱₜ / (1 - ᾱₜ)
        snr = alphas_bar / (1 - alphas_bar)
        self.register("snr", snr)

        # reverse process 需要的係數
        alphas_bar_prev = np.append(1.0, alphas_bar[:-1])  # ᾱₜ₋₁
        self.register("alphas_bar_prev",   alphas_bar_prev)

        # posterior variance σₜ²
        posterior_var = betas * (1 - alphas_bar_prev) / (1 - alphas_bar)
        posterior_var = np.clip(posterior_var, 1e-20, None)
        self.register("posterior_var",     posterior_var)
        self.register("posterior_std",     np.sqrt(posterior_var))

    def register(self, name, arr):
        setattr(self, name, torch.tensor(arr, dtype=torch.float32))

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_bar",
            "sqrt_alphas_bar", "sqrt_one_minus_ab",
            "alphas_bar_prev", "posterior_var", "posterior_std",
            "snr",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ── helper：根據 t index 取值並 reshape 成 (B,1,1,1) ──
    def _gather(self, arr, t):
        # arr: (T,)  t: (B,)
        # 回傳 (B, 1, 1, 1)，方便和 image tensor broadcast
        return arr[t].reshape(-1, 1, 1, 1)

    def v_to_eps(self, xt, v, t):
        sqrt_ab   = self._gather(self.sqrt_alphas_bar,   t)
        sqrt_1mab = self._gather(self.sqrt_one_minus_ab, t)
        return sqrt_1mab * xt + sqrt_ab * v

    def v_to_x0(self, xt, v, t):
        sqrt_ab   = self._gather(self.sqrt_alphas_bar,   t)
        sqrt_1mab = self._gather(self.sqrt_one_minus_ab, t)
        return sqrt_ab * xt - sqrt_1mab * v

    def get_v_target(self, x0, noise, t):
        sqrt_ab   = self._gather(self.sqrt_alphas_bar,   t)
        sqrt_1mab = self._gather(self.sqrt_one_minus_ab, t)
        return sqrt_ab * noise - sqrt_1mab * x0

    # ─────────────────────────────────────────
    # Forward process：x₀ → xₜ（訓練時用）
    # ─────────────────────────────────────────

    def q_sample(self, x0, t, noise=None):
        """
        給定乾淨圖片 x0 和 timestep t，直接一步加 noise 得到 xₜ。
        xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab   = self._gather(self.sqrt_alphas_bar,   t)  # √ᾱₜ
        sqrt_1mab = self._gather(self.sqrt_one_minus_ab, t)  # √(1-ᾱₜ)

        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    # ─────────────────────────────────────────
    # Reverse process：xₜ → xₜ₋₁（inference 時用）
    # ─────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, model, xt, t, cond, eos_idx, cfg_scale=1.0):
        """
        給定 xₜ，用 model 預測 noise，還原一步得到 xₜ₋₁。
        """
        t_tensor = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)

        # model 預測 noise
        eps_pred = model(xt, t_tensor, cond, eos_idx)
        if cfg_scale > 1.0:
            uncond = torch.zeros_like(cond)
            eps_uncond = model(xt, t_tensor, uncond, eos_idx)
            eps_pred = eps_uncond + cfg_scale * (eps_pred - eps_uncond)

        # 計算係數
        alpha     = self._gather(self.alphas,            t_tensor)  # αₜ
        sqrt_1mab = self._gather(self.sqrt_one_minus_ab, t_tensor)  # √(1-ᾱₜ)
        post_std  = self._gather(self.posterior_std,     t_tensor)  # σₜ

        # xₜ₋₁ = mean + noise（t=0 時不加 noise）
        mean = (1 / alpha.sqrt()) * (xt - (1 - alpha) / sqrt_1mab * eps_pred)
        if t == 0:
            return mean
        noise = torch.randn_like(xt)
        return mean + post_std * noise

    @torch.no_grad()
    def sample(self, model, shape, cond, eos_idx, device, cfg_scale=1.0):
        """
        從純 noise 開始，反覆去噪 T 步，生成圖片。

        shape:   (B, 3, H, W)
        cond:    (B, 77, 512)
        eos_idx: (B,) long tensor
        """
        x = torch.randn(shape, device=device)   # xₜ ~ N(0, I)

        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond, eos_idx, cfg_scale=cfg_scale)

        return x   # x₀，範圍約 [-1, 1]

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, eos_idx, device,
                    ddim_steps=50, eta=0.0, cfg_scale=1.0, prediction_type="v",
                    initial_noise=None):
        """
        cond:          (B, 77, 512)
        eos_idx:       (B,) long tensor
        initial_noise: (B, C, H, W) float32, optional — caller can supply
                       per-image noise for reproducibility / diversity control.
        """
        step_size = self.T // ddim_steps
        timesteps = torch.arange(0, self.T, step_size, dtype=torch.long).flip(0)

        x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
        uncond = torch.zeros_like(cond)

        for i, t in enumerate(timesteps):
            t_val = t.item()
            t_tensor = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

            pred = model(x, t_tensor, cond, eos_idx)
            if cfg_scale > 1.0:
                pred_uncond = model(x, t_tensor, uncond, eos_idx)
                pred = pred_uncond + cfg_scale * (pred - pred_uncond)

            if prediction_type == "v":
                x0_pred  = self.v_to_x0(x, pred, t_tensor)
                eps_pred = self.v_to_eps(x, pred, t_tensor)
            else:
                sqrt_ab   = self._gather(self.sqrt_alphas_bar,   t_tensor)
                sqrt_1mab = self._gather(self.sqrt_one_minus_ab, t_tensor)
                x0_pred  = (x - sqrt_1mab * pred) / sqrt_ab
                eps_pred = pred

            # Dynamic thresholding (Imagen §2.4): prevents saturation at high CFG scales
            # by clipping to the 99.5th percentile instead of hard [-1,1]
            if cfg_scale > 1.0:
                s = x0_pred.abs().flatten(1).quantile(0.995, dim=1)
                s = s.clamp(min=1.0).view(-1, 1, 1, 1)
                x0_pred = x0_pred.clamp(-s, s) / s
            else:
                x0_pred = x0_pred.clamp(-1, 1)

            if i == len(timesteps) - 1:
                x = x0_pred
                break

            t_next = timesteps[i + 1].item()
            t_next_tensor = torch.full((shape[0],), t_next, device=device, dtype=torch.long)
            sqrt_ab        = self._gather(self.sqrt_alphas_bar,   t_tensor)
            sqrt_1mab      = self._gather(self.sqrt_one_minus_ab, t_tensor)
            sqrt_ab_next   = self._gather(self.sqrt_alphas_bar,   t_next_tensor)
            sqrt_1mab_next = self._gather(self.sqrt_one_minus_ab, t_next_tensor)

            sigma = eta * (sqrt_1mab_next / sqrt_1mab) * \
                    (1 - sqrt_ab ** 2 / sqrt_ab_next ** 2).sqrt()
            direction = (sqrt_1mab_next ** 2 - sigma ** 2).clamp(min=0).sqrt() * eps_pred
            noise = torch.randn_like(x) if eta > 0 else 0
            x = sqrt_ab_next * x0_pred + direction + sigma * noise

        return x


# ─────────────────────────────────────────────
# 快速測試
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from model import UNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2

    scheduler = DDPMScheduler(T=1000).to(device)

    # 確認 schedule 的數值範圍合理
    print(f"β  range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"ᾱ  range: [{scheduler.alphas_bar.min():.4f}, {scheduler.alphas_bar.max():.4f}]")
    print(f"ᾱ  at t=0:   {scheduler.alphas_bar[0]:.4f}  (應接近 1.0)")
    print(f"ᾱ  at t=999: {scheduler.alphas_bar[999]:.4f}  (應接近 0.0)")
    print()

    # Forward process 測試
    model   = UNet(base_ch=96, ch_mult=(1,2,4,4), num_res=3, time_dim=576).to(device)
    x0      = torch.randn(B, 3, 64, 64).to(device)
    t       = torch.randint(0, 1000, (B,)).to(device)
    cond    = torch.randn(B, 77, 512).to(device)
    eos_idx = torch.randint(5, 10, (B,)).to(device)

    xt, noise = scheduler.q_sample(x0, t)
    print(f"q_sample input  x0: {x0.shape}  range [{x0.min():.2f}, {x0.max():.2f}]")
    print(f"q_sample output xt: {xt.shape}  range [{xt.min():.2f}, {xt.max():.2f}]")
    print()

    # UNet forward（training step）
    v_pred = model(xt, t, cond, eos_idx)
    v_target = scheduler.get_v_target(x0, noise, t)
    loss = torch.nn.functional.mse_loss(v_pred, v_target)
    print(f"v_pred: {v_pred.shape}")
    print(f"loss:   {loss.item():.4f}  (初始應接近 1.0)")