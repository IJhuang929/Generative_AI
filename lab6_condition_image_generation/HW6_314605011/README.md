# HW6 — Conditional Image Generation（314605011）

## 繳交內容

```
HW6_314605011/
├── generated_images/       # 2,000 張 64×64 PNG（已生成）
├── scripts/
│   ├── train.py            # 訓練腳本
│   ├── generate.py         # 生成腳本
│   └── data/
│       └── generate.csv    # 生成用 CSV（已附）
├── model.pth               # 用於生成的模型權重（Epoch 750 EMA）
├── train_filtered.csv      # 訓練用 CSV（已附，4,408 筆）
├── README.md
└── requirements.txt
```

---

## 環境設定

**Python 版本：3.10+**（建議 3.11 或 3.13）

```bash
# 建立虛擬環境並安裝套件（以下擇一）

# 方法一：pip
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 方法二：uv（更快）
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **CUDA**：需要 NVIDIA GPU + CUDA 12.x。無 GPU 時腳本會自動切至 CPU（速度極慢）。

---

## 復現生成圖片

`generated_images/` 已包含完整 2,000 張圖片，若要重新生成請執行：

**步驟 1**：確認目錄結構（`model.pth` 與 `scripts/` 在同一層）

```
HW6_314605011/
├── model.pth
└── scripts/
    ├── generate.py
    └── data/
        └── generate.csv   ← 已附
```

**步驟 2**：切換至 `scripts/` 並執行

```bash
cd HW6_314605011/scripts
python generate.py
```

生成完成後，2,000 張 PNG 會輸出至 `HW6_314605011/generated_images/`。

> 單張圖片生成時間約 2–3 秒（RTX 3090）；2,000 張全跑約 10 分鐘。

### 生成參數

| 參數 | 值 |
|------|----|
| Sampler | DDIM（確定性，eta=0） |
| DDIM steps | 200 |
| CFG scale | 3.5 |
| Batch size | 32 |

---

## 訓練方式

> 訓練圖片（`trainset/`）未附於繳交包，需自行下載。`train_filtered.csv` 已附。

**步驟 1**：下載訓練圖片並放入 `trainset/`

訓練集來源：Kaggle [brainrot-image-dataset](https://www.kaggle.com/datasets/otto29/brainrot-image-dataset)

下載後，確認目錄如下：

```
HW6_314605011/
├── scripts/
│   └── train.py
├── trainset/               ← 放入從 Kaggle 下載的訓練圖片（.png）
└── train_filtered.csv      ← 已附（4,408 筆）
```

**步驟 2**（選用）：設定 Weights & Biases

```bash
wandb login          # 需要 W&B 帳號
# 或關閉 W&B 紀錄：
export WANDB_MODE=offline
```

**步驟 3**：開始訓練

```bash
cd HW6_314605011
python scripts/train.py \
    --csv train_filtered.csv \
    --img_dir trainset \
    --save_dir checkpoints
```

Checkpoint 每 50 epoch 存一次至 `checkpoints/unet_epoch_<N>.pth`（套用 EMA 權重）。  
本次提交使用 **Epoch 750** 的 EMA 權重，訓練於 Kaggle NVIDIA Tesla P100（16 GB）。

---

## 模型架構

自行實作的 **SimpleUNet**（noise-prediction DDPM + CFG + EMA）：

| 階段 | 模組 |
|------|------|
| Encoder | 3 個 ResBlock + strided Conv 下採樣；第 2、3 層加 CrossAttention |
| Bottleneck | ResBlock → SelfAttention → ResBlock |
| Decoder | 3 個 ConvTranspose 上採樣 + ResBlock；第 1、2 層加 CrossAttention |

- **文字條件**：CLIP `ViT-B-32-quickgelu` token-level 輸出（77×512），透過 CrossAttention 注入 UNet
- **CFG**：訓練時 10% 機率以空字串 embedding 替換條件；推論 `cfg_scale=3.5`
- **EMA**：decay=0.9999，每 50 epoch 套用後存檔

### 訓練超參數

| 參數 | 值 |
|------|----|
| batch_size | 64 |
| epochs | 1000（使用 ep.750） |
| learning_rate | 2e-4 |
| lr schedule | Linear warmup 5% + Cosine decay |
| optimizer | AdamW（weight_decay=1e-4） |
| DDPM timesteps | 1000（linear β: 1e-4 → 0.02） |
| CFG drop rate | 0.1 |
| EMA decay | 0.9999 |
| precision | AMP float16 |

---

## 評分結果

使用 Codabench 官方 `score.py` 本地評估：

| 指標 | 分數 |
|------|------|
| FID | 351.6 |
| CLIP-T | 0.270 |
