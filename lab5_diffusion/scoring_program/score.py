# Some libraries and options
import argparse
import glob
import json
import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import open_clip
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from tqdm import tqdm
from scipy import linalg


class ImageSizeError(ValueError):
    pass


class FIDDataset(Dataset):
    def __init__(self, image_paths: str, image_size: int = None):
        self.image_paths = image_paths
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.image_size is not None and img.size != (self.image_size, self.image_size):
            raise ImageSizeError(
                f"Image {img_path} has size {img.size}, expected {self.image_size}x{self.image_size}"
            )

        if self.transform:
            img = self.transform(img)
        return img


class CLIPDataset(Dataset):
    def __init__(self, image_paths: str, items: List[Tuple[str, str]], preprocess, tokenizer, image_size: int = None):
        self.image_paths = image_paths
        self.items = items
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_size = image_size

        if len(image_paths) != len(items):
            raise ValueError(
                f"Number of image paths ({len(image_paths)}) does not match number of items ({len(items)})"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.image_size is not None and img.size != (self.image_size, self.image_size):
            raise ImageSizeError(
                f"Image {img_path} has size {img.size}, expected {self.image_size}x{self.image_size}"
            )
        img = self.preprocess(img)

        text_prompt = self.items[idx]["text_prompt"]
        tokens = self.tokenizer(text_prompt)[0]

        return img, tokens


def get_inception_model() -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    return model


def get_inception_feature(dataloader: DataLoader, model: nn.Module, device: torch.device, verbose: bool = False) -> torch.Tensor:
    features_list = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            ncols=0,
            desc="Extracting features",
            leave=False,
            disable=not verbose
        ):
            batch = batch.to(device)
            features = model(batch)
            features_list.append(features.cpu())
    return torch.cat(features_list, axis=0)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


class CLIPScore:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_image_embedding(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_embedding(self, tokens):
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def clip_score_image_image(self, images1, images2):
        embs1 = self.get_image_embedding(images1)
        embs2 = self.get_image_embedding(images2)
        scores = (embs1 * embs2).sum(dim=-1)
        return scores

    def clip_score_image_text(self, images, tokens):
        embsi = self.get_image_embedding(images)
        embst = self.get_text_embedding(tokens)
        scores = (embsi * embst).sum(dim=-1)
        return scores


def run_fid(args, ref_dir, res_dir, device):
    # ================ Load reference statistics ================
    ref_mu_path = os.path.join(ref_dir, args.ref_mu)
    ref_sigma_path = os.path.join(ref_dir, args.ref_sigma)
    real_mu = np.load(ref_mu_path)
    real_sigma = np.load(ref_sigma_path)

    # ================ List and check images ================
    image_paths = glob.glob(os.path.join(res_dir, '**', '*.png'), recursive=True)
    if args.num_images is not None and len(image_paths) != args.num_images:
        raise ValueError(
            f"Expected {args.num_images} PNG files in zip file, "
            f"found {len(image_paths)}.")
    if args.verbose:
        print(f"[INFO] Found {len(image_paths)} images in the zip file.")

    # ================ Load and process images =================
    dataset = FIDDataset(image_paths, args.image_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    # Initialize the Inception model and extract features
    model = get_inception_model().to(device)
    if args.verbose:
        print("[INFO] Generating Inception model features...")
    features = get_inception_feature(dataloader, model, device, verbose=args.verbose)
    del model
    features = features.numpy()
    fake_mu = np.mean(features, axis=0)
    fake_sigma = np.cov(features, rowvar=False)

    # ================ Calculate FID =================
    if args.verbose:
        print("[INFO] Calculating FID score...")
    fid = calculate_fid(real_mu, real_sigma, fake_mu, fake_sigma)
    if args.verbose:
        print(f"[INFO] FID score: {fid:.4f}.")
    return {"FID": fid}


def run_clip(args, ref_dir, res_dir, device):
    if not ('clip_i' in args.scores or 'clip_t' in args.scores):
        raise ValueError(
            "CLIP scores can only be calculated if 'clip_i' or 'clip_t' is in "
            "the --scores argument."
        )

    # ================ Load reference statistics ================
    test_json_path = os.path.join(ref_dir, args.test_json)
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test JSON file {test_json_path} does not exist.")
    if args.verbose:
        print(f"[INFO] Loading test JSON file: {test_json_path}.")
    with open(test_json_path, 'r') as f:
        items = json.load(f)
    items = [
        {
            "image_name": item["image_name"],
            "text_prompt": item["text_prompt"]
        } for item in items.values()
    ]

    # ================ List and check images ================
    image_paths = glob.glob(os.path.join(res_dir, '**', '*.png'), recursive=True)
    image_base2path = {os.path.basename(path): path for path in image_paths}
    if len(image_paths) != len(items):
        raise ValueError(
            f"Expected {len(items)} PNG files in zip file, found "
            f"{len(image_paths)}.")
    if args.num_images is None:
        num_images = len(items)
    else:
        num_images = args.num_images
    if len(image_base2path) != num_images:
        raise ValueError(
            f"The number of unique image filenames ({len(image_base2path)}) "
            f"does not match the number of items in the test JSON file ({num_images})."
        )
    if args.verbose:
        print(f"[INFO] Found {len(image_paths)} images in the zip file.")

    if args.verbose:
        print(
            "[INFO] Checking if all image paths from the test JSON file are "
            "present in the zip file.")
    for item in items:
        # item["image_name"] is the base filename of the image
        if item["image_name"] not in image_base2path:
            raise ValueError(
                f"Image path {item['image_name']} from the test JSON file "
                f"not found in the zip file."
            )

    # ================ Load CLIP model ================
    if args.verbose:
        print(f"[INFO] Loading CLIP model: {args.model_name} with pretrained weights: {args.pretrained}.")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    clip_score = CLIPScore(model, tokenizer)

    # ================ Load and process images =================
    dataset = CLIPDataset(image_paths, items, preprocess, tokenizer, image_size=args.image_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    if args.verbose:
        print(f"[INFO] Calculating CLIP scores for {len(dataset)} images...")
    scores_list = defaultdict(list)
    for image, tokens in tqdm(
        dataloader,
        desc="Calculating CLIP scores",
        ncols=0,
        disable=not args.verbose,
        leave=False,
    ):
        image = image.to(device)
        tokens = tokens.to(device)
        for score in args.scores:
            if score == 'clip_i':
                scores = clip_score.clip_score_image_image(image, image)
            elif score == 'clip_t':
                scores = clip_score.clip_score_image_text(image, tokens)
            else:
                scores = None
            if scores is not None:
                scores_list[score].append(scores.cpu())
    for key in scores_list:
        scores_list[key] = torch.cat(scores_list[key], dim=0).mean().item()

    # ================ Save the score =================
    output_json = {}
    for score in args.scores:
        if score == 'clip_i':
            score_name = "CLIP Image-Image Score"
        elif score == 'clip_t':
            score_name = "CLIP Image-Text Score"
        else:
            score_name = None
        if score_name is not None:
            if args.verbose:
                print(f"[INFO] {score_name}: {scores_list[score]:.6f}")
            output_json[score_name] = scores_list[score]

    return output_json


def main():
    parser = argparse.ArgumentParser(
        description='Scoring program for the Image Generation Competition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        default="input",
        help=(
            'Input directory containing the `res` directory with the images '
            'and `ref` directory with the testing resources.'
        )
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        default="output",
        help='Output directory where the scores will be saved.'
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help=(
            "Size of the images in the zip file. If None, no check is performed. "
            "This is used to ensure that all images are of the same size."
        )
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help=(
            "Check if the number of images in the zip file matches this "
            "number. If None, the number of images is checked only when "
            "--test_json is provided and `clip_i` or `clip_t` scores are "
            "calculated."
        )
    )
    parser.add_argument(
        "--scores",
        type=str,
        nargs='+',
        choices=['fid', 'clip_i', 'clip_t'],
        default=['fid'],
        help=(
            "Type of scores to calculate. `fid` for FID score, `clip_i` for "
            "CLIP Image-Image Score and `clip_t` for CLIP Text-Image Score."
        )
    )
    parser.add_argument(
        "--ref_mu",
        type=str,
        default="test_mu.npy",
        help=(
            "Path to the inception feature mean for the reference dataset. "
            "This is relative to the ${input_dir}/ref directory."
        )
    )
    parser.add_argument(
        "--ref_sigma",
        type=str,
        default="test_sigma.npy",
        help=(
            "Path to the inception feature covariance for the reference dataset. "
            "This is relative to the ${input_dir}/ref directory."
        )
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="test.json",
        help=(
            "Path to the test JSON file containing image paths and text prompts. "
            "This is relative to the ${input_dir}/ref directory."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="`model_name` to use for the `open_clip.create_model_and_transforms` function."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="`pretrained` to use for the `open_clip.create_model_and_transforms` function."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inception feature extraction"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=(
            'Path to the configuration file in JSON format. This is relative '
            'to the ${input_dir}/ref directory. If provided, it will override '
            'the parameters set in the command line.'
        )
    )
    args = parser.parse_args()
    device = torch.device("cuda:0")

    # ================ Set up directories and paths =================
    ref_dir = os.path.join(args.input_dir, 'ref')   # Reference directory. Contains the statistics
    res_dir = os.path.join(args.input_dir, 'res')   # Results directory. Submitted predictions
    if args.verbose:
        print(f"[INFO] Reference directory: {ref_dir}.")
        print(f"[INFO] Results directory: {res_dir}.")

    # ================ Load configuration if provided =================
    if args.config is not None:
        config_path = os.path.join(ref_dir, args.config)
        if args.verbose:
            print(f"[INFO] Loading configuration from {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
        with open(config_path, 'r') as f:
            config = json.load(f)
        for key in args.__dict__:
            if key in config:
                setattr(args, key, config[key])
                del config[key]
        if args.verbose and len(config) > 0:
            print(f"[WARNING] The following parameters were not found in the command line: {config.keys()}")
    else:
        if args.verbose:
            print("[INFO] No configuration file provided, using command line arguments.")

    # ================ Start calculating scores =================
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        if args.verbose:
            print("[INFO] Starting CLIP score calculation.")
        clip_scores_json = run_clip(args, ref_dir, res_dir, device)
        output_json.update(clip_scores_json)

    if 'fid' in args.scores:
        if args.verbose:
            print("[INFO] Starting FID score calculation.")
        fid_json = run_fid(args, ref_dir, res_dir, device)
        output_json.update(fid_json)

    output_file = os.path.join(args.output_dir, 'scores.json')
    if args.verbose:
        print(f"[INFO] Saving scores to: {output_file}")
        print(f"[INFO] The content of the output file: {json.dumps(output_json)}")
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    main()
