import torch
import open_clip
import pandas as pd
from PIL import Image
from tqdm import tqdm


IMG_DIR   = "trainset"
CSV_IN    = "train.csv"
CSV_OUT   = "train_filtered.csv"
THRESHOLD = 0.20


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model = model.to(device).eval()

    df = pd.read_csv(CSV_IN)
    scores = []

    with torch.no_grad():
        for row in tqdm(df.itertuples(), total=len(df), desc="Scoring"):
            img = preprocess(
                Image.open(f"{IMG_DIR}/{row.id}").convert("RGB")
            ).unsqueeze(0).to(device)

            prompt = f"a {row.animal} and a {row.object}"
            tokens = tokenizer([prompt]).to(device)

            img_emb = model.encode_image(img)
            txt_emb = model.encode_text(tokens)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            scores.append((img_emb @ txt_emb.T).item())

    df["clip_score"] = scores

    # ── 分數分布 ──────────────────────────────────────
    print("\n=== CLIP score distribution ===")
    print(df["clip_score"].describe().round(4))

    percentiles = [5, 10, 15, 20, 25, 30]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"  {p:3d}%  →  {df['clip_score'].quantile(p/100):.4f}")

    print("\nImages kept at various thresholds:")
    for t in [0.15, 0.18, 0.20, 0.22, 0.25]:
        kept = (df["clip_score"] >= t).sum()
        print(f"  threshold {t:.2f}  →  {kept}/{len(df)} images ({kept/len(df)*100:.1f}%)")

    # ── per-combo 最少剩多少張 ─────────────────────────
    filtered = df[df["clip_score"] >= THRESHOLD]
    combo_counts = filtered.groupby(["animal", "object"]).size()
    print(f"\n=== After filtering (threshold={THRESHOLD}) ===")
    print(f"Total images kept: {len(filtered)}/{len(df)} ({len(filtered)/len(df)*100:.1f}%)")
    print(f"Images per combo — min: {combo_counts.min()}  median: {combo_counts.median():.0f}  max: {combo_counts.max()}")
    print("\nCombos with fewest images:")
    print(combo_counts.sort_values().head(10).to_string())

    # ── 存檔 ──────────────────────────────────────────
    filtered[["id", "animal", "object"]].to_csv(CSV_OUT, index=False)
    print(f"\nSaved {CSV_OUT}  ({len(filtered)} rows)")

    # 同時存含分數的版本供事後分析
    df.to_csv("train_with_scores.csv", index=False)
    print(f"Saved train_with_scores.csv  (all {len(df)} rows with scores)")


if __name__ == "__main__":
    main()
