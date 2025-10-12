#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from utils import tqdm_, probs2class, save_images

def load_gray_unit_tensor(p: Path) -> torch.Tensor:
    """
    Load a grayscale image and convert to 1xHxW float32 in [0,1],
    mirroring img_transform in main.py.
    """
    img = Image.open(p).convert("L")
    arr = np.asarray(img, dtype=np.float32)[None, ...] / 255.0
    return torch.from_numpy(arr)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices_dir", type=str, required=True,
                        help="Folder with 2D test slices (e.g., data/SEGTHOR_TEST/test/img)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output folder for per-slice predicted masks (PNG)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pickled torch model (bestmodel.pkl)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes (K). Used only for label scaling to PNG.")
    args = parser.parse_args()

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # Load the whole pickled model (torch.save(model, ...))
    print(f"[INFO] Loading model: {args.model_path}")
    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    # Match the same label scaling used elsewhere in your code:
    # gt_transform uses 63 for K==5, else 255/(K-1).
    K = int(args.num_classes)
    mult = 63 if K == 5 else int(round(255.0 / (K - 1)))

    # Collect images
    slices_dir = Path(args.slices_dir)
    if not slices_dir.exists():
        raise FileNotFoundError(f"slices_dir not found: {slices_dir}")
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([p for p in slices_dir.rglob("*") if p.suffix.lower() in img_exts])
    if not files:
        raise RuntimeError(f"No images found under {slices_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_tensors = []
    batch_paths = []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)  # Bx1xHxW

        # Forward pass — handle deep supervision tuples
        out = model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        probs = F.softmax(logits, dim=1)  # BxKxHxW
        pred_classes = probs2class(probs)  # BxHxW (from utils)
        # Scale for PNG encoding
        scaled = (pred_classes * mult).to(torch.uint8)  # BxHxW

        # Save with your project helper (utils.save_images)
        # save_images expects: segs (Tensor), names (Iterable[str]), root (Path)
        names = [p.stem for p in batch_paths]
        save_images(scaled, names, out_dir)

        batch_tensors.clear()
        batch_paths.clear()

    for p in tqdm_(files, desc="[Infer]"):
        batch_tensors.append(load_gray_unit_tensor(p))
        batch_paths.append(p)
        if len(batch_tensors) >= args.batch_size:
            flush()

    flush()
    print(f"[OK] Saved per-slice predictions to: {out_dir}")

if __name__ == "__main__":
    main()
