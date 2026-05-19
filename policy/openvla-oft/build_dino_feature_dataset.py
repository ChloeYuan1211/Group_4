"""
build_dino_feature_dataset.py

Construct the DINO feature dataset as specified in DINO_feature_dataset_spec.md.

For each frame extracted from video episodes:
  1. Apply 5 geometric transforms: up/down/left/right shift + horizontal flip
  2. Run each through DINOv2 to get 16x16xD patch features
  3. Reverse-align features to original patch coordinates
  4. Compute masked average => F^GT
  5. Save (image, feature_gt, valid_count)

Optimized for A100 80GB: batches N_frames * 5 transforms into a single GPU
forward pass with bf16 inference. Uses background thread for async saving.

Usage (conda env 'openvla'):
    python build_dino_feature_dataset.py \
        --checkpoint_dir /path/to/checkpoint \
        --data_root /path/to/aloha-agilex_randomized_500 \
        --output_dir ./dino_feature_dataset \
        --shift_k 1 \
        --num_frames_per_batch 32
"""

import argparse
import os
import sys
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import partial

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F_torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. Build DINOv2 featurizer (reused from visualize_dino_attention.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_dinov2(image_size: int = 224):
    """Create a DINOv2-ViT-L model via timm, matching openvla-oft config."""
    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=image_size,
    )
    model.eval()
    return model


def load_finetuned_weights(model, checkpoint_path: str):
    """Load fine-tuned DINOv2 weights from openvla-oft's vision_backbone checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    prefix = "vision_backbone.featurizer."
    dino_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            dino_state[k[len(prefix):]] = v

    lora_alpha = 16
    lora_r = 32
    scaling = lora_alpha / lora_r

    base_weights = {}
    lora_a = {}
    lora_b = {}
    plain_weights = {}

    for k, v in dino_state.items():
        parts = k.split(".")
        if len(parts) >= 2 and parts[1] in ("scale", "shift"):
            continue

        if ".base_layer." in k:
            param_path = k.replace(".base_layer.", ".").replace(".block.", ".")
            base_weights[param_path] = v
        elif ".lora_A.default." in k:
            param_path = k.split(".lora_A.default.")[0].replace(".block.", ".")
            suffix = k.split(".lora_A.default.")[1]
            lora_a[param_path + "." + suffix] = v
        elif ".lora_B.default." in k:
            param_path = k.split(".lora_B.default.")[0].replace(".block.", ".")
            suffix = k.split(".lora_B.default.")[1]
            lora_b[param_path + "." + suffix] = v
        else:
            mapped_key = k.replace(".block.", ".")
            mapped_key = mapped_key.replace(".scale_factor", ".gamma")
            plain_weights[mapped_key] = v

    merged = dict(base_weights)
    for key in list(merged.keys()):
        if key in lora_a and key in lora_b:
            a = lora_a[key]
            b = lora_b[key]
            w = merged[key]
            if a.dim() == 2 and b.dim() == 2:
                merged[key] = w + scaling * (b @ a)
            elif a.dim() == 4 and b.dim() == 4:
                a_flat = a.reshape(a.shape[0], -1)
                b_flat = b.reshape(b.shape[0], -1)
                delta = (b_flat @ a_flat).reshape(w.shape)
                merged[key] = w + scaling * delta

    merged.update(plain_weights)

    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print("[INFO] All DINOv2 weights loaded perfectly.")
    else:
        print("[INFO] Fine-tuned DINOv2 weights loaded (with minor key mismatches).")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 2. Image preprocessing (matching openvla-oft's DINOv2 transform)
# ──────────────────────────────────────────────────────────────────────────────

def get_dino_transform(model, image_size: int = 224):
    """Build the same preprocessing pipeline openvla-oft uses for DINOv2."""
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, image_size, image_size)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    assert isinstance(transform, Compose)
    assert isinstance(transform.transforms[0], Resize)
    target_size = (image_size, image_size)
    transform = Compose([
        Resize(target_size, interpolation=transform.transforms[0].interpolation),
        *transform.transforms[1:],
    ])
    return transform


# ──────────────────────────────────────────────────────────────────────────────
# 3. DINO patch feature extraction (batched)
# ──────────────────────────────────────────────────────────────────────────────

def extract_patch_features_batch(model, img_batch, device, image_size=224, patch_size=14):
    """
    Extract patch-level features from DINOv2 for a BATCH of images.

    timm's get_intermediate_layers already returns patch tokens only
    (CLS and register tokens are stripped internally), so we must NOT
    do any additional prefix-token slicing.

    Args:
        model: DINOv2 model
        img_batch: (B, 3, 224, 224) tensor
        device: torch device

    Returns:
        patch_features: (B, 16, 16, 1024) tensor on CPU
    """
    num_patches_per_side = image_size // patch_size  # 16
    num_patch_tokens = num_patches_per_side ** 2      # 256

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        layer_idx = len(model.blocks) - 2
        features = model.get_intermediate_layers(img_batch.to(device), n={layer_idx})
        patch_tokens = features[0]  # (B, 256, 1024) — already patch-only

    # Hard assertions: token count and embedding dim must be exactly right
    B = patch_tokens.shape[0]
    assert patch_tokens.shape[1] == num_patch_tokens, (
        f"Expected {num_patch_tokens} patch tokens, got {patch_tokens.shape[1]}. "
        f"get_intermediate_layers may have changed its return format."
    )
    assert patch_tokens.shape[2] == model.embed_dim, (
        f"Expected embed_dim={model.embed_dim}, got {patch_tokens.shape[2]}."
    )

    # Reshape to spatial grid: (B, 16, 16, 1024)
    patch_features = patch_tokens.reshape(B, num_patches_per_side, num_patches_per_side, -1)
    return patch_features.float().cpu()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Geometric transforms on PIL images
# ──────────────────────────────────────────────────────────────────────────────

def apply_shift(pil_img, direction, k, patch_size=14):
    """
    Apply a pixel-level shift to a PIL image.

    Shift amount = k * patch_size pixels.
    The region that slides out is cropped; the new region is filled with white (255).

    Args:
        pil_img: PIL Image (224x224)
        direction: one of 'up', 'down', 'left', 'right'
        k: number of patches to shift
        patch_size: pixel size of one patch (14 for 224/16)

    Returns:
        shifted PIL Image (224x224)
    """
    img = np.array(pil_img)  # (H, W, 3)
    H, W = img.shape[:2]
    shift_px = k * patch_size
    result = np.full_like(img, 255)  # white canvas

    if direction == 'right':
        # Image content moves right: right side cropped, left side filled white
        result[:, shift_px:, :] = img[:, :W - shift_px, :]
    elif direction == 'left':
        # Image content moves left: left side cropped, right side filled white
        result[:, :W - shift_px, :] = img[:, shift_px:, :]
    elif direction == 'down':
        # Image content moves down: bottom cropped, top filled white
        result[shift_px:, :, :] = img[:H - shift_px, :, :]
    elif direction == 'up':
        # Image content moves up: top cropped, bottom filled white
        result[:H - shift_px, :, :] = img[shift_px:, :, :]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return Image.fromarray(result)


def apply_hflip(pil_img):
    """Apply horizontal flip to a PIL image."""
    return pil_img.transpose(Image.FLIP_LEFT_RIGHT)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Feature reverse alignment & mask construction
# ──────────────────────────────────────────────────────────────────────────────

def reverse_align_shift(features, direction, k, grid_size=16):
    """
    Reverse-align shifted features back to original patch coordinates.

    When image was shifted right by k patches:
      - Feature is shifted LEFT by k patches to align back
      - Right k columns become invalid

    Args:
        features: (16, 16, D) tensor
        direction: the direction the IMAGE was shifted
        k: number of patches shifted
        grid_size: patch grid size (16)

    Returns:
        aligned_features: (16, 16, D) tensor (invalid regions filled with 0)
        mask: (16, 16) tensor with 1=valid, 0=invalid
    """
    D = features.shape[2]
    aligned = torch.zeros(grid_size, grid_size, D)
    mask = torch.ones(grid_size, grid_size)

    if direction == 'right':
        # Image shifted right => feature shift LEFT to align
        # Right k columns become invalid
        aligned[:, :grid_size - k, :] = features[:, k:, :]
        mask[:, grid_size - k:] = 0
    elif direction == 'left':
        # Image shifted left => feature shift RIGHT to align
        # Left k columns become invalid
        aligned[:, k:, :] = features[:, :grid_size - k, :]
        mask[:, :k] = 0
    elif direction == 'down':
        # Image shifted down => feature shift UP to align
        # Bottom k rows become invalid
        aligned[:grid_size - k, :, :] = features[k:, :, :]
        mask[grid_size - k:, :] = 0
    elif direction == 'up':
        # Image shifted up => feature shift DOWN to align
        # Top k rows become invalid
        aligned[k:, :, :] = features[:grid_size - k, :, :]
        mask[:k, :] = 0
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return aligned, mask


def reverse_align_hflip(features):
    """
    Reverse-align horizontally flipped features.
    Simply flip the feature map back along the width axis.
    All positions are valid.

    Args:
        features: (16, 16, D) tensor

    Returns:
        aligned_features: (16, 16, D) tensor
        mask: (16, 16) tensor (all ones)
    """
    aligned = torch.flip(features, dims=[1])  # flip along width
    mask = torch.ones(features.shape[0], features.shape[1])
    return aligned, mask


# ──────────────────────────────────────────────────────────────────────────────
# 6. Core: compute F^GT for a BATCH of images
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_one_image(args_tuple):
    """Worker: apply all 5 transforms to a single image. Used by ThreadPoolExecutor."""
    pil_img, transform, transforms_spec, patch_size = args_tuple
    tensors = []
    for transform_type, k in transforms_spec:
        if transform_type == 'hflip':
            transformed_img = apply_hflip(pil_img)
        else:
            transformed_img = apply_shift(pil_img, transform_type, k, patch_size)
        tensors.append(transform(transformed_img))
    return tensors


def prepare_transforms_batch(pil_images, transform, shift_k, patch_size=14,
                             num_workers=8):
    """
    Prepare all 5 geometric transform versions for a batch of PIL images.
    Returns a single stacked tensor ready for batched DINO inference.

    Each image is processed in parallel via ThreadPoolExecutor (same logic,
    PIL/numpy ops release the GIL so threads run concurrently).

    Args:
        pil_images: list of N PIL images (224x224)
        transform: DINO preprocessing transform
        shift_k: number of patches to shift
        patch_size: pixel size of one patch
        num_workers: number of parallel CPU threads

    Returns:
        batch_tensor: (N*5, 3, 224, 224) tensor
        transforms_order: list of (transform_type, k) for alignment
    """
    transforms_spec = [
        ('up', shift_k),
        ('down', shift_k),
        ('left', shift_k),
        ('right', shift_k),
        ('hflip', None),
    ]

    work_items = [(img, transform, transforms_spec, patch_size) for img in pil_images]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        per_image_tensors = list(executor.map(_prepare_one_image, work_items))

    # Flatten: [img0_t0, img0_t1, ..., img1_t0, img1_t1, ...]
    all_tensors = [t for img_tensors in per_image_tensors for t in img_tensors]

    batch_tensor = torch.stack(all_tensors, dim=0)  # (N*5, 3, 224, 224)
    return batch_tensor, transforms_spec


def compute_feature_gt_batch(model, transform, pil_images, device, shift_k=1,
                             image_size=224, patch_size=14, grid_size=16):
    """
    Compute F^GT for a BATCH of images in a single GPU forward pass.

    Args:
        model: DINOv2 model
        transform: image preprocessing transform
        pil_images: list of N PIL images (224x224)
        device: torch device
        shift_k: number of patches to shift

    Returns:
        features_gt: list of N tensors, each (16, 16, D_dino)
        valid_counts: list of N tensors, each (16, 16)
    """
    N = len(pil_images)
    num_transforms = 5

    # Prepare all transformed images as a single batch
    batch_tensor, transforms_spec = prepare_transforms_batch(
        pil_images, transform, shift_k, patch_size
    )  # (N*5, 3, 224, 224)

    # Single batched forward pass through DINO
    all_features = extract_patch_features_batch(
        model, batch_tensor, device, image_size, patch_size
    )  # (N*5, 16, 16, 1024)

    # Verify embedding dimension is exactly 1024 (DINOv2 ViT-L)
    D = all_features.shape[-1]
    assert D == 1024, (
        f"DINO feature dim must be 1024, got {D}. "
        f"Check extract_patch_features_batch for token extraction errors."
    )
    all_features = all_features.reshape(N, num_transforms, grid_size, grid_size, D)

    # Pre-compute masks (same for all images with same shift_k)
    # These only depend on the transform type, not the image content
    features_gt = []
    valid_counts = []

    for i in range(N):
        aligned_features = []
        masks = []

        for t_idx, (transform_type, k) in enumerate(transforms_spec):
            feat = all_features[i, t_idx]  # (16, 16, D)

            if transform_type == 'hflip':
                aligned, mask = reverse_align_hflip(feat)
            else:
                aligned, mask = reverse_align_shift(feat, transform_type, k, grid_size)

            aligned_features.append(aligned)
            masks.append(mask)

        aligned_stack = torch.stack(aligned_features, dim=0)  # (5, 16, 16, D)
        mask_stack = torch.stack(masks, dim=0)                 # (5, 16, 16)

        mask_expanded = mask_stack.unsqueeze(-1)  # (5, 16, 16, 1)
        numerator = (mask_expanded * aligned_stack).sum(dim=0)
        denominator = mask_stack.sum(dim=0).unsqueeze(-1)

        feature_gt = numerator / denominator.clamp(min=1.0)
        valid_count = mask_stack.sum(dim=0)

        features_gt.append(feature_gt)
        valid_counts.append(valid_count)

    return features_gt, valid_counts


# ──────────────────────────────────────────────────────────────────────────────
# 7. Video frame extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_all_frames(video_path: str):
    """Extract ALL frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Cannot read video: {video_path}")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_idx += 1
    cap.release()
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# 8. Async saver: background thread for I/O
# ──────────────────────────────────────────────────────────────────────────────

class AsyncSaver:
    """Background thread that saves .pt files so GPU doesn't wait on disk I/O."""

    def __init__(self, num_workers=2):
        self._queue = queue.Queue(maxsize=256)
        self._workers = []
        self._count = 0
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            path, data = item
            torch.save(data, path)
            self._queue.task_done()

    def save(self, path, data):
        self._queue.put((path, data))
        self._count += 1

    def flush(self):
        self._queue.join()

    def shutdown(self):
        self.flush()
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            t.join()

    @property
    def count(self):
        return self._count


# ──────────────────────────────────────────────────────────────────────────────
# 9. Main: build the dataset (batched + parallel)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build DINO feature dataset (F^GT) from video episodes"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/runs/train_full_singletask/"
                "openvla-7b+aloha_move_can_pot+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--100000_chkpt",
        help="Path to the merged openvla-oft checkpoint directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/qdhe/workspace/dda4210/project/dataset/data_raw/move_can_pot/aloha-agilex_randomized_500",
        help="Root directory of the dataset (containing video/ subfolder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_feature_dataset",
        help="Directory to save the constructed dataset",
    )
    parser.add_argument("--shift_k", type=int, default=1,
                        help="Number of patch-grid units to shift (default: 1 = 14px)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size for DINOv2")
    parser.add_argument("--num_frames_per_batch", type=int, default=32,
                        help="Number of frames to batch together (each produces 5 transforms). "
                             "Total GPU batch = num_frames_per_batch * 5. Default 32 -> 160 images/batch.")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="Starting episode index (for resuming)")
    parser.add_argument("--end_episode", type=int, default=-1,
                        help="Ending episode index (exclusive, -1 for all)")
    parser.add_argument("--save_workers", type=int, default=4,
                        help="Number of background threads for saving .pt files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Shift k = {args.shift_k} patches = {args.shift_k * 14} pixels")
    print(f"[INFO] Frames per batch: {args.num_frames_per_batch} -> "
          f"{args.num_frames_per_batch * 5} images per GPU forward pass")

    # ── Build DINOv2 model ──
    print("[INFO] Building DINOv2 ViT-L model...")
    model = build_dinov2(image_size=args.image_size)

    # ── Load fine-tuned weights ──
    vision_ckpt = os.path.join(args.checkpoint_dir, "vision_backbone--100000_checkpoint.pt")
    if os.path.exists(vision_ckpt):
        print(f"[INFO] Loading fine-tuned vision weights from {vision_ckpt}")
        model = load_finetuned_weights(model, vision_ckpt)
    else:
        print(f"[WARN] No vision backbone checkpoint found at {vision_ckpt}; using pretrained weights.")

    model = model.to(device)
    model.eval()

    # ── Build transform (must be done before compile, needs model data_cfg) ──
    transform = get_dino_transform(model, image_size=args.image_size)

    # ── Discover video episodes ──
    video_dir = os.path.join(args.data_root, "video")
    if not os.path.isdir(video_dir):
        print(f"[ERROR] Video directory not found: {video_dir}")
        sys.exit(1)

    video_files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith(".mp4")],
        key=lambda x: int(x.replace("episode", "").replace(".mp4", ""))
    )
    print(f"[INFO] Found {len(video_files)} video episodes in {video_dir}")

    # Handle episode range
    if args.end_episode == -1:
        args.end_episode = len(video_files)
    video_files = video_files[args.start_episode:args.end_episode]
    print(f"[INFO] Processing episodes {args.start_episode} to {args.start_episode + len(video_files) - 1}")

    # ── Create output directory ──
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Process each episode ──
    patch_size = 14
    grid_size = args.image_size // patch_size  # 16
    to_tensor = transforms.ToTensor()

    # Metadata for the dataset
    metadata = {
        "image_size": args.image_size,
        "patch_size": patch_size,
        "grid_size": grid_size,
        "shift_k": args.shift_k,
        "checkpoint": args.checkpoint_dir,
        "data_root": args.data_root,
        "transforms": ["up", "down", "left", "right", "hflip"],
        "num_frames_per_batch": args.num_frames_per_batch,
    }

    # Save metadata
    import json
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata to {meta_path}")

    # Start async saver
    saver = AsyncSaver(num_workers=args.save_workers)

    total_samples = 0
    NFB = args.num_frames_per_batch

    # ── Episode prefetch: background thread decodes + resizes next episode ──
    # This overlaps video I/O with GPU compute so the GPU never waits for decode.
    prefetch_queue = queue.Queue(maxsize=2)

    def _prefetch_worker(vid_files_to_fetch, image_size_):
        for vf in vid_files_to_fetch:
            ep_name = vf.replace(".mp4", "")
            ep_idx = int(ep_name.replace("episode", ""))
            vpath = os.path.join(video_dir, vf)
            ep_dir = os.path.join(args.output_dir, ep_name)
            done_m = os.path.join(ep_dir, "_done")
            if os.path.exists(done_m):
                prefetch_queue.put((ep_name, ep_idx, ep_dir, done_m, None, None))
                continue
            try:
                raw_frames = extract_all_frames(vpath)
            except Exception as exc:
                prefetch_queue.put((ep_name, ep_idx, ep_dir, done_m, exc, None))
                continue
            resized = [
                f.resize((image_size_, image_size_), Image.BILINEAR)
                for f in raw_frames
            ]
            del raw_frames
            img_tensors = [to_tensor(img) for img in resized]
            prefetch_queue.put((ep_name, ep_idx, ep_dir, done_m, None, (resized, img_tensors)))
        prefetch_queue.put(None)  # sentinel

    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(video_files, args.image_size),
        daemon=True,
    )
    prefetch_thread.start()

    for vid_file in tqdm(video_files, desc="Episodes"):
        item = prefetch_queue.get()
        if item is None:
            break

        episode_name, episode_idx, episode_dir, done_marker, err, payload = item
        os.makedirs(episode_dir, exist_ok=True)

        if os.path.exists(done_marker):
            continue

        if err is not None:
            print(f"  [ERROR] Failed to extract frames: {err}")
            continue

        resized_frames, image_tensors = payload
        num_frames = len(resized_frames)

        # Filter out already-processed frames for resuming
        frame_indices = [
            fi for fi in range(num_frames)
            if not os.path.exists(os.path.join(episode_dir, f"frame_{fi:05d}.pt"))
        ]

        if not frame_indices:
            with open(done_marker, "w") as f:
                f.write("done\n")
            continue

        # Process in batches of NFB frames
        for batch_start in range(0, len(frame_indices), NFB):
            batch_fi = frame_indices[batch_start:batch_start + NFB]
            batch_pil = [resized_frames[fi] for fi in batch_fi]

            # Batched F^GT computation: one GPU forward pass for len(batch_fi)*5 images
            features_gt, valid_counts = compute_feature_gt_batch(
                model, transform, batch_pil, device,
                shift_k=args.shift_k,
                image_size=args.image_size,
                patch_size=patch_size,
                grid_size=grid_size,
            )

            # Queue saves to background threads
            for j, fi in enumerate(batch_fi):
                sample = {
                    "image": image_tensors[fi],
                    "feature_gt": features_gt[j],
                    "valid_count": valid_counts[j],
                    "episode": episode_idx,
                    "frame": fi,
                }
                saver.save(os.path.join(episode_dir, f"frame_{fi:05d}.pt"), sample)
                total_samples += 1

        del resized_frames, image_tensors

        # Mark episode as done (after saver flushes this episode's saves)
        saver.flush()
        with open(done_marker, "w") as f:
            f.write("done\n")

    prefetch_thread.join()
    # Shutdown saver
    saver.shutdown()

    print(f"\n[INFO] Dataset construction complete!")
    print(f"[INFO] Total samples saved: {total_samples}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Feature shape: ({grid_size}, {grid_size}, D_dino)")
    print(f"[INFO] Each sample contains: image, feature_gt, valid_count, episode, frame")


if __name__ == "__main__":
    main()
