import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import open_clip
from tqdm import tqdm

VAE_SCALE = 0.18215   # SD VAE latent scaling factor


class BrainrotDataset(Dataset):
    def __init__(self, csv_path, img_dir, image_size=64, device="cuda",
                 latent_mode=False, min_per_combo=30):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.device    = device
        self.latent_mode = latent_mode

        if not latent_mode:
            # Balanced sampling: duplicate rows of under-represented combos
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

        if latent_mode:
            cache_path        = os.path.splitext(csv_path)[0] + "_latents.pt"
            self.latent_cache = self._precompute_latents(cache_path, image_size)

    def _precompute_clip_embeddings(self):
        print("Precomputing CLIP embeddings (token-level + EOS index)...")
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        model = model.to(self.device).eval()

        prompts = {
            f"a {row.animal} and a {row.object}"
            for row in self.df.itertuples()
        }

        cache = {}
        with torch.no_grad():
            for prompt in prompts:
                tokens = tokenizer([prompt]).to(self.device)
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
        sample = next(iter(cache.values()))
        print(f"Cached {len(cache)} prompts, tokens: {sample['tokens'].shape}, eos sample: {sample['eos_idx']}")
        return cache

    def _precompute_latents(self, cache_path, image_size):
        if os.path.exists(cache_path):
            print(f"Loading latent cache from {cache_path}...")
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "cache" in data:
                # new format: includes normalization stats
                self.latent_mean = data["mean"]   # (4,)
                self.latent_std  = data["std"]    # (4,)
                return data["cache"]
            # old format (no stats): delete and regenerate
            print("Old cache format detected, regenerating with normalization...")
            import os as _os; _os.remove(cache_path)

        print("Encoding images to VAE latents (one-time)...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # [0,1] → [-1,1]
        ])

        latent_cache = {}
        with torch.no_grad():
            for row in tqdm(self.df.itertuples(), total=len(self.df), desc="VAE encode"):
                img   = Image.open(os.path.join(self.img_dir, row.id)).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(self.device)        # (1, 3, H, W)
                latent = vae.encode(img_t).latent_dist.mode() * VAE_SCALE  # (1, 4, 8, 8)
                latent_cache[row.id] = latent.squeeze(0).cpu()             # (4, 8, 8)

        del vae
        torch.cuda.empty_cache()

        # per-channel normalize：讓每個 channel mean=0, std=1
        all_latents = torch.stack(list(latent_cache.values()))  # (N, 4, 8, 8)
        self.latent_mean = all_latents.mean(dim=(0, 2, 3))      # (4,)
        self.latent_std  = all_latents.std(dim=(0, 2, 3))       # (4,)
        for k in latent_cache:
            latent_cache[k] = (latent_cache[k] - self.latent_mean[:, None, None]) \
                               / self.latent_std[:, None, None]

        torch.save({"cache": latent_cache,
                    "mean":  self.latent_mean,
                    "std":   self.latent_std}, cache_path)
        print(f"Saved latent cache ({len(latent_cache)} images) → {cache_path}")
        print(f"  latent_mean={self.latent_mean.tolist()}")
        print(f"  latent_std= {self.latent_std.tolist()}")
        return latent_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        prompt = f"a {row['animal']} and a {row['object']}"
        entry   = self.clip_cache[prompt]
        tokens  = entry["tokens"]                       # (77, 512)
        eos_idx = entry["eos_idx"]                      # int

        if self.latent_mode:
            latent = self.latent_cache[row["id"]].clone()
            if torch.rand(1).item() < 0.5:
                latent = latent.flip(-1)
            return latent, tokens, eos_idx

        img = Image.open(os.path.join(self.img_dir, row["id"])).convert("RGB")
        return self.transform(img), tokens, eos_idx     # (3, H, W), (77, 512), int


if __name__ == "__main__":
    ds = BrainrotDataset(
        csv_path   = "train_filtered.csv",
        img_dir    = "trainset",
        device     = "cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Dataset size: {len(ds)}")
    img, tokens, eos_idx = ds[0]
    print(f"Image shape  : {img.shape}   dtype={img.dtype}")
    print(f"Image range  : [{img.min():.2f}, {img.max():.2f}]")
    print(f"tokens: {tokens.shape}   dtype={tokens.dtype}")
    print(f"eos_idx: {eos_idx}")
