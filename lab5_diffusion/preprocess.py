"""
preprocess.py
=============
使用 face-crop-plus 對教授臉部照片做對齊、裁切、品質增強前處理。

用法:
    uv run python preprocess.py
    uv run python preprocess.py --input_dir images --output_dir images_aligned
    uv run python preprocess.py --no_enhance   # 跳過品質增強（比較快）
    uv run python preprocess.py --dry_run      # 只處理前 10 張，確認效果
"""

import os
import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Face alignment and quality enhancement preprocessing.")
    parser.add_argument("--input_dir",  type=str, default="images",
                        help="原始訓練圖片資料夾")
    parser.add_argument("--output_dir", type=str, default="images_aligned",
                        help="處理後圖片輸出資料夾")
    parser.add_argument("--output_size", type=int, default=256,
                        help="輸出圖片尺寸（正方形）")
    parser.add_argument("--face_factor", type=float, default=0.7,
                        help="臉部佔圖片的比例（0~1），越大臉越大")
    parser.add_argument("--no_enhance", action="store_true",
                        help="跳過品質增強，只做對齊裁切（速度快 3x）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只處理前 10 張，確認效果後再跑全部")
    parser.add_argument("--device", type=str, default="cuda",
                        help="使用的裝置（cuda 或 cpu）")
    return parser.parse_args()


def dry_run_setup(input_dir: str, dry_run_dir: str, n: int = 10):
    """複製前 n 張圖片到暫存目錄，用於 dry run。"""
    os.makedirs(dry_run_dir, exist_ok=True)
    images = sorted(Path(input_dir).glob("*.png"))[:n]
    if not images:
        images = sorted(Path(input_dir).glob("*.jpg"))[:n]
    for img in images:
        shutil.copy(img, dry_run_dir)
    print(f"[Dry Run] 複製 {len(images)} 張圖片到 {dry_run_dir}")
    return dry_run_dir


def count_images(directory: str) -> int:
    d = Path(directory)
    return len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))


def main():
    args = parse_args()

    # ========== 安裝檢查 ==========
    try:
        from face_crop_plus import Cropper
    except ImportError:
        print("[ERROR] 找不到 face-crop-plus，請先安裝：")
        print("    uv add face-crop-plus")
        print("    或 pip install face-crop-plus")
        exit(1)

    # ========== Dry Run 設定 ==========
    input_dir = args.input_dir
    if args.dry_run:
        dry_run_dir = "/tmp/hw5_preprocess_dryrun"
        input_dir = dry_run_setup(args.input_dir, dry_run_dir)
        output_dir = "/tmp/hw5_preprocess_dryrun_output"
        print(f"[Dry Run] 輸出到 {output_dir}，確認效果後刪除")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ========== 統計輸入圖片 ==========
    total_input = count_images(input_dir)
    print(f"輸入圖片：{total_input} 張（{input_dir}）")
    print(f"輸出目錄：{output_dir}")
    print(f"輸出尺寸：{args.output_size}×{args.output_size}")
    print(f"臉部比例：{args.face_factor}")
    print(f"品質增強：{'關閉' if args.no_enhance else '開啟（只增強模糊圖片）'}")
    print(f"裝置：{args.device}")
    print()

    # ========== 建立 Cropper ==========
    cropper = Cropper(
        output_size=args.output_size,
        face_factor=args.face_factor,

        # 每張圖只取最大的臉（教授照片通常只有一張臉）
        strategy="largest",

        # 偵測閾值：0.999 過濾掉模糊/背景臉，只保留清晰的正面臉
        det_threshold=0.999,

        # 品質增強：只對臉部比例很小的圖片做增強（避免浪費時間）
        # 設為 None 表示完全關閉，設為 0.01 表示只增強臉很小的圖
        enh_threshold=None if args.no_enhance else 0.01,

        # 邊緣填充方式：replicate 比 constant（黑邊）更自然
        padding="replicate",

        # 使用 GPU 加速
        device=args.device,
    )

    # ========== 執行前處理 ==========
    print("開始前處理...")
    cropper.process_dir(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    # ========== 統計結果 ==========
    total_output = count_images(output_dir)
    dropped = total_input - total_output
    print()
    print("=" * 50)
    print(f"前處理完成！")
    print(f"  輸入：{total_input} 張")
    print(f"  輸出：{total_output} 張")
    print(f"  過濾掉：{dropped} 張（偵測不到臉、側臉、模糊）")
    print(f"  保留率：{total_output/total_input*100:.1f}%")
    print()

    if args.dry_run:
        print(f"[Dry Run] 確認結果在 {output_dir}")
        print(f"[Dry Run] 滿意的話執行完整前處理：")
        print(f"    uv run python preprocess.py")
    else:
        print(f"接下來把 train.py 的 image_dir 改成：")
        print(f'    image_dir = "{output_dir}"')

    print("=" * 50)


if __name__ == "__main__":
    main()