import os
import json
import shutil
import argparse
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

from model import UNet
from diffusion import DDPMScheduler


# ─────────────────────────────────────────────
# CLIP encoding
# ─────────────────────────────────────────────

def build_clip_cache(prompts, device):
    print("Encoding prompts with CLIP (token-level + EOS)...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model = model.to(device).eval()

    cache = {}
    with torch.no_grad():
        for prompt in prompts:
            if prompt in cache:
                continue
            tokens = tokenizer([prompt]).to(device)
            x = model.token_embedding(tokens)
            x = x + model.positional_embedding
            x = model.transformer(x, attn_mask=model.attn_mask)
            x = model.ln_final(x)                          # (1, 77, 512)
            eos_idx = tokens.argmax(dim=-1).item()
            cache[prompt] = {
                "tokens":  x.squeeze(0).cpu().float(),
                "eos_idx": eos_idx,
            }

    del model
    torch.cuda.empty_cache()
    print(f"Cached {len(cache)} prompts.")
    return cache


# ─────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed fixed to {args.seed} (visualization mode)")
    else:
        print("No seed fixed (evaluation mode)")

    # ── 讀 generate.csv ──
    df = pd.read_csv(args.generate_csv)
    df["prompt"] = "a " + df["animal"] + " and a " + df["object"]

    # ── 建立輸出目錄 ──
    res_dir = os.path.join(args.input_dir, "res")
    ref_dir = os.path.join(args.input_dir, "ref")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 複製 ref 檔案 ──
    for fname in ["test_mu.npy", "test_sigma.npy", "config.json"]:
        src = os.path.join(args.ref_src, fname)
        dst = os.path.join(ref_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    # ── 產生 test.json ──
    test_json = [
        {"image_name": row["id"], "text_prompt": row["prompt"]}
        for _, row in df.iterrows()
    ]
    test_json_path = os.path.join(ref_dir, "test.json")
    with open(test_json_path, "w") as f:
        json.dump(test_json, f, indent=2)
    print(f"Saved test.json ({len(test_json)} items) → {test_json_path}")

    # ── 複製真實圖片到 ref/test/ （CLIP-I 用）──
    test_img_dir = os.path.join(ref_dir, "test")
    os.makedirs(test_img_dir, exist_ok=True)
    train_df = pd.read_csv(args.train_csv)
    train_df["prompt"] = "a " + train_df["animal"] + " and a " + train_df["object"]
    pair2img = {}
    for _, row in train_df.iterrows():
        if row["prompt"] not in pair2img:
            pair2img[row["prompt"]] = row["id"]

    for _, row in df.iterrows():
        real_id  = pair2img.get(row["prompt"])
        dst_path = os.path.join(test_img_dir, row["id"])
        if real_id and not os.path.exists(dst_path):
            src_path = os.path.join(args.img_dir, real_id)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

    print(f"Copied real images → {test_img_dir}")

    # ── 載入模型 ──
    print(f"Loading model from {args.ckpt}...")
    model = UNet(
        in_ch    = 3,
        base_ch  = 96,
        ch_mult  = (1, 2, 4, 4),
        num_res  = 3,
        time_dim = 576,
        dropout  = 0.10,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    
    key = "ema_model" if "ema_model" in ckpt else "model"
    # 🚀 如果你的舊 checkpoint ema_model 壞了(全是雜訊)，請取消註解下面這行強制使用 model：
    # key = "model"
    
    model.load_state_dict(ckpt[key])
    model = model.half()
    model.eval()
    print(f"Loaded checkpoint (step {ckpt.get('step', '?')}, weights={key})")

    scheduler = DDPMScheduler(T=1000).to(device)

    # ── 預計算 CLIP embeddings ──
    unique_prompts = df["prompt"].unique().tolist()
    clip_cache = build_clip_cache(unique_prompts, device)

    # ── 生成圖片（按 prompt 分組 batch 生成）──
    print("Generating images...")
    grouped = df.groupby("prompt")

    for prompt, group in tqdm(grouped, desc="Prompts"):
        ids     = group["id"].tolist()
        entry   = clip_cache[prompt]
        cond    = entry["tokens"].unsqueeze(0).to(device).half()              # (1, 77, 512)
        eos_idx = torch.tensor([entry["eos_idx"]], device=device, dtype=torch.long)

        # 為每張圖片用 image_id 的 hash 當 seed，確保：
        #   1. 同 prompt 的不同圖片有獨立的 noise（diversity）
        #   2. 相同 id 每次執行都生出一樣的圖（reproducibility）
        noises = []
        for img_id in ids:
            if args.seed is not None:
                g = torch.Generator()
                g.manual_seed(args.seed + int(img_id.split(".")[0]))
                noises.append(torch.randn(3, 64, 64, generator=g))
            else:
                noises.append(torch.randn(3, 64, 64))
        all_noise = torch.stack(noises)                      # (N, 3, 64, 64)

        generated = []
        offset = 0
        while offset < len(ids):
            batch = min(args.gen_batch, len(ids) - offset)
            cond_batch  = cond.expand(batch, -1, -1)         # (batch, 77, 512)
            eos_batch   = eos_idx.expand(batch)              # (batch,)
            noise_batch = all_noise[offset:offset + batch]   # (batch, 3, 64, 64)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                imgs = scheduler.ddim_sample(
                    model           = model,
                    shape           = (batch, 3, 64, 64),
                    cond            = cond_batch,
                    eos_idx         = eos_batch,
                    device          = device,
                    ddim_steps      = args.ddim_steps,
                    eta             = args.eta,
                    cfg_scale       = args.cfg_scale,
                    prediction_type = "v",
                    initial_noise   = noise_batch,
                )
            imgs = imgs.float()

            generated.append(imgs.cpu())
            offset += batch

        generated = torch.cat(generated, dim=0)             # (N, 3, 64, 64)

        # 存成 PNG
        for img_tensor, img_id in zip(generated, ids):
            img = (img_tensor.clamp(-1, 1) + 1) / 2        # [-1,1] → [0,1]
            img = (img * 255).byte().permute(1, 2, 0).numpy()
            Image.fromarray(img).save(os.path.join(res_dir, img_id))

    print(f"\nGenerated {len(df)} images → {res_dir}  [DDIM v-pred, {args.ddim_steps} steps]")
    print("\nTo score, run:")
    print(f"  python score.py --input_dir {args.input_dir} --output_dir {args.output_dir} --scores fid clip_t clip_i --config config.json --verbose")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         type=str, required=True,
                        help="path to checkpoint .pth file")
    parser.add_argument("--generate_csv", type=str, default="generate.csv")
    parser.add_argument("--train_csv",    type=str, default="train.csv")
    parser.add_argument("--img_dir",      type=str, default="trainset",
                        help="real training images (for CLIP-I)")
    parser.add_argument("--ref_src",      type=str, default=".",
                        help="directory containing test_mu.npy, test_sigma.npy, config.json")
    parser.add_argument("--input_dir",    type=str, default="input",
                        help="score.py input directory")
    parser.add_argument("--output_dir",   type=str, default="output",
                        help="score.py output directory")
    parser.add_argument("--gen_batch",    type=int,   default=10)
    parser.add_argument("--sampler",      type=str,   default="ddim", choices=["ddim"])
    parser.add_argument("--ddim_steps",   type=int,   default=100)
    parser.add_argument("--eta",          type=float, default=0.0)
    parser.add_argument("--cfg_scale",    type=float, default=5.0)
    parser.add_argument("--seed",         type=int,   default=None,
                        help="評估 FID 時不要傳 seed，視覺化才用")
    args = parser.parse_args()

    generate(args)
