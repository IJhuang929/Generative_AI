"""
pipeline.py
-----------
依序執行：train.py → generate.py → score.py
每個階段結束後釋放 GPU 記憶體。

用法：
    uv run python pipeline.py --ckpt_dir checkpoints --max_steps 100000 --batch_size 32
    uv run python pipeline.py --skip_train --ckpt checkpoints/20260611_184859/ckpt_100000.pth
"""

import os
import re
import zipfile
import gc
import sys
import json
import glob
import argparse
import subprocess
import torch


# ─────────────────────────────────────────────
# GPU cleanup
# ─────────────────────────────────────────────

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[pipeline] GPU memory freed.\n")


# ─────────────────────────────────────────────
# 分數計算
# ─────────────────────────────────────────────

def grade_clip_t(score):
    if score >= 0.2703:
        return "100%  (≥ baseline4)"
    elif score >= 0.2618:
        return "90%   (≥ baseline3)"
    elif score >= 0.2536:
        return "80%   (≥ baseline2)"
    elif score >= 0.2170:
        return "70%   (≥ baseline1)"
    else:
        return "0%    (below baseline1)"

def grade_fid(score):
    if score <= 49.2545:
        return "100%  (≤ baseline4)"
    elif score <= 58.0755:
        return "90%   (≤ baseline3)"
    elif score <= 75.0642:
        return "80%   (≤ baseline2)"
    elif score <= 90.0142:
        return "70%   (≤ baseline1)"
    else:
        return "0%    (above baseline1)"


# ─────────────────────────────────────────────
# 找最新 checkpoint
# ─────────────────────────────────────────────

def find_latest_ckpt(ckpt_dir):
    """在 checkpoints/ 下找最新的 run，回傳最後一個 .pth 路徑"""
    runs = sorted(glob.glob(os.path.join(ckpt_dir, "*")))
    if not runs:
        raise FileNotFoundError(f"No runs found in {ckpt_dir}")
    latest_run = runs[-1]
    pths = sorted(glob.glob(os.path.join(latest_run, "ckpt_*.pth")))
    if not pths:
        raise FileNotFoundError(f"No .pth files found in {latest_run}")
    return pths[-1]


# ─────────────────────────────────────────────
# 執行子程序
# ─────────────────────────────────────────────

def run(cmd, tag):
    print(f"\n{'='*60}")
    print(f"[pipeline] Starting: {tag}")
    print(f"[pipeline] Command : {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=True)
    return result



# ─────────────────────────────────────────────
# 打包繳交
# ─────────────────────────────────────────────

def pack_submission(res_dir, fid, clip_t):
    """
    把 res/ 下的圖片壓縮成 result_FID{fid:.1f}_CLIPT{clip_t:.4f}.zip
    只有 FID 和 CLIP-T 都過 baseline1（70%）才打包。
    """
    fid_pass    = fid    <= 90.0142
    clip_t_pass = clip_t >= 0.2170

    if not (fid_pass and clip_t_pass):
        print("[pipeline] Scores below baseline1, skipping pack.")
        return

    zip_name = f"result_FID{fid:.1f}_CLIPT{clip_t:.4f}.zip"
    print(f"[pipeline] Packing submission → {zip_name}")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(res_dir)):
            if fname.endswith(".png"):
                zf.write(os.path.join(res_dir, fname), arcname=fname)

    size_mb = os.path.getsize(zip_name) / 1e6
    print(f"[pipeline] Done: {zip_name}  ({size_mb:.1f} MB, {len(zf.namelist())} images)")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main(args):
    python = sys.executable

    # ── Step 1: Train ──
    if not args.skip_train:
        train_cmd = [
            python, "train.py",
            "--max_steps",    str(args.max_steps),
            "--batch_size",   str(args.batch_size),
            "--num_workers",  str(args.num_workers),
            "--lr",           str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--save_every",   str(args.save_every),
            "--ckpt_dir",     args.ckpt_dir,
        ]
        if args.resume:
            train_cmd += ["--resume", args.resume]
        run(train_cmd, "TRAIN")
        free_gpu()
    else:
        print("[pipeline] Skipping training (--skip_train)")

    # ── Step 2: Find checkpoint ──
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = find_latest_ckpt(args.ckpt_dir)
    print(f"[pipeline] Using checkpoint: {ckpt_path}\n")

    # ── Step 3: Generate ──
    gen_cmd = [
        python, "generate.py",
        "--ckpt",         ckpt_path,
        "--generate_csv", args.generate_csv,
        "--train_csv",    args.train_csv,
        "--img_dir",      args.img_dir,
        "--input_dir",    args.input_dir,
        "--output_dir",   args.output_dir,
        "--gen_batch",    str(args.gen_batch),
        "--sampler",      args.sampler,
        "--ddim_steps",   str(args.ddim_steps),
        "--eta",          str(args.eta),
        "--cfg_scale",    str(args.cfg_scale),
    ]
    run(gen_cmd, "GENERATE")
    free_gpu()

    # ── Step 4: Score ──
    os.makedirs(args.output_dir, exist_ok=True)
    m = re.search(r'ckpt_(\d+)\.pth', ckpt_path)
    ckpt_epoch = int(m.group(1)) if m else None
    score_cmd = [
        python, "score.py",
        "--input_dir",  args.input_dir,
        "--output_dir", args.output_dir,
        "--scores", "fid", "clip_t", "clip_i",
        "--config", "config.json",
        "--verbose",
        "--cfg_scale",  str(args.cfg_scale),
        "--sampler",    args.sampler,
        "--ddim_steps", str(args.ddim_steps),
    ]
    if ckpt_epoch is not None:
        score_cmd += ["--epoch", str(ckpt_epoch)]
    run(score_cmd, "SCORE")
    free_gpu()

    # ── Step 5: 讀分數並印出成績 ──
    scores_path = os.path.join(args.output_dir, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)
        if isinstance(scores, list):
            scores = scores[-1]

        fid    = scores.get("FID",    None)
        clip_t = scores.get("CLIP_T", None)
        clip_i = scores.get("CLIP_I", None)

        print("\n" + "="*60)
        print("  RESULTS")
        print("="*60)
        if fid is not None:
            print(f"  FID    : {fid:.4f}  →  {grade_fid(fid)}")
        if clip_t is not None:
            print(f"  CLIP-T : {clip_t:.4f}  →  {grade_clip_t(clip_t)}")
        if clip_i is not None:
            print(f"  CLIP-I : {clip_i:.4f}")
        print("="*60 + "\n")

        # 打包繳交
        if fid is not None and clip_t is not None:
            res_dir = os.path.join(args.input_dir, "res")
            pack_submission(res_dir, fid, clip_t)
    else:
        print(f"[pipeline] scores.json not found at {scores_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument("--skip_train",   action="store_true",
                        help="skip training, go straight to generate")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="checkpoint path to resume training from")
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints")
    parser.add_argument("--max_steps",    type=int,   default=100_000)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--save_every",   type=int,   default=5000)

    # generate
    parser.add_argument("--ckpt",         type=str,   default=None,
                        help="explicit checkpoint path; if omitted, uses latest")
    parser.add_argument("--generate_csv", type=str,   default="generate.csv")
    parser.add_argument("--train_csv",    type=str,   default="train.csv")
    parser.add_argument("--img_dir",      type=str,   default="trainset")
    parser.add_argument("--input_dir",    type=str,   default="input")
    parser.add_argument("--output_dir",   type=str,   default="output")
    parser.add_argument("--gen_batch",    type=int,   default=200)
    parser.add_argument("--sampler",      type=str,   default="ddim",
                        choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps",   type=int,   default=100)
    parser.add_argument("--eta",          type=float, default=0.0)
    parser.add_argument("--cfg_scale",    type=float, default=5.0,
                        help="Classifier-Free Guidance scale (1.0=off, 7.5=recommended)")

    args = parser.parse_args()
    main(args)
