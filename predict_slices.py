#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# Mirror your main.py normalization
def img_to_tensor_gray_unit(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to 1xHxW float32 in [0,1].
    """
    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)[None, ...] / 255.0
    return torch.from_numpy(arr)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices_dir", type=str, required=True,
                        help="Folder with 2D test slices (PNG/JPG). Expected names like Patient_01_0001.png")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output folder for per-slice predicted masks (PNG)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pickled torch model (bestmodel.pkl)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=5,
                        help="K (used only to choose scaling factor for PNG labels)")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Load whole pickled model (saved via torch.save(model, ...))
    print(f"[INFO] Loading model from: {args.model_path}")
    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    # Decide scaling factor to keep compatibility with your training/stitching
    K = args.num_classes
    mult = 63 if K == 5 else int(round(255.0 / (K - 1)))

    # Gather slice files
    slices_dir = Path(args.slices_dir)
    if not slices_dir.exists():
        raise FileNotFoundError(f"slices_dir not found: {slices_dir}")
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted([p for p in slices_dir.rglob("*") if p.suffix.lower() in img_exts])
    if not files:
        raise RuntimeError(f"No images found under {slices_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simple batched inference
    batch = []
    batch_paths = []

    def flush_batch(batch_tensors, batch_paths_local):
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)  # Bx1xHxW
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]  # in case of deep supervision models
        probs = F.softmax(logits, dim=1)  # BxKxHxW
        preds = probs.argmax(dim=1)       # BxHxW

        for pred, path in zip(preds, batch_paths_local):
            mask = (pred.cpu().numpy().astype(np.uint8)) * mult
            # Preserve flat output structure mirroring inputs
            out_path = out_dir / path.name
            Image.fromarray(mask).save(out_path)

    for p in tqdm(files, desc="[Infer]"):
        tensor = img_to_tensor_gray_unit(Image.open(p))
        batch.append(tensor)
        batch_paths.append(p)
        if len(batch) >= args.batch_size:
            flush_batch(batch, batch_paths)
            batch.clear()
            batch_paths.clear()

    # Flush remaining
    flush_batch(batch, batch_paths)

    print(f"[OK] Saved per-slice predictions to: {out_dir}")
    print("     (Note: files keep the same names as inputs; "
          "e.g., Patient_01_0001.png -> predicted PNG in out_dir)")

if __name__ == "__main__":
    main()
