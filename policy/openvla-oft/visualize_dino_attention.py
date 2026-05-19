"""
visualize_dino_attention.py

Visualize DINOv2 self-attention maps from OpenVLA-OFT's fused vision backbone.
Extracts frames from video episodes, runs them through the fine-tuned DINOv2,
and produces attention heatmap overlays.

Usage (from conda env 'openvla'):
    python visualize_dino_attention.py \
        --checkpoint_dir /path/to/openvla-oft/checkpoint \
        --data_root /path/to/dataset/data_raw/move_can_pot \
        --output_dir ./dino_attention_vis \
        --num_frames 10
"""

import argparse
import os
import sys
from pathlib import Path
from functools import partial

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.models.vision_transformer import Attention
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# 1. Build DINOv2 featurizer (same way openvla-oft does)
# ──────────────────────────────────────────────────────────────────────────────

def build_dinov2(image_size: int = 224):
    """Create a DINOv2-ViT-L model via timm, matching openvla-oft config.
    Uses pretrained=False since we load weights from the fine-tuned checkpoint."""
    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=image_size,
    )
    model.eval()
    return model


def load_finetuned_weights(model, checkpoint_path: str):
    """
    Load fine-tuned DINOv2 weights from openvla-oft's vision_backbone checkpoint.
    Handles LoRA-merged weight keys and the nested block structure.

    Checkpoint key patterns (after removing 'vision_backbone.featurizer.' prefix):
      blocks.X.block.attn.qkv.base_layer.weight  -> blocks.X.attn.qkv.weight (+ LoRA merge)
      blocks.X.block.ls1.scale_factor             -> blocks.X.ls1.gamma
      blocks.X.block.norm1.weight                 -> blocks.X.norm1.weight
      patch_embed.proj.base_layer.weight           -> patch_embed.proj.weight (+ LoRA merge)
      blocks.X.scale.*, blocks.X.shift.*           -> skip (FiLM wrappers, not in timm)
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Filter only DINOv2 (featurizer) keys, remove prefix
    prefix = "vision_backbone.featurizer."
    dino_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            dino_state[k[len(prefix):]] = v

    # LoRA config
    lora_alpha = 16
    lora_r = 32
    scaling = lora_alpha / lora_r

    # Categorize keys
    base_weights = {}   # param_path -> tensor (from .base_layer.)
    lora_a = {}         # param_path -> tensor
    lora_b = {}         # param_path -> tensor
    plain_weights = {}  # direct weights (no LoRA, e.g., norm, cls_token)
    skip_keys = set()

    for k, v in dino_state.items():
        # Skip FiLM scale/shift wrappers (not in timm model)
        parts = k.split(".")
        if len(parts) >= 2 and parts[1] in ("scale", "shift"):
            skip_keys.add(k)
            continue

        if ".base_layer." in k:
            # e.g. blocks.0.block.attn.qkv.base_layer.weight
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
            # Plain weight: just remove .block. nesting
            mapped_key = k.replace(".block.", ".")
            # Rename scale_factor -> gamma for LayerScale
            mapped_key = mapped_key.replace(".scale_factor", ".gamma")
            plain_weights[mapped_key] = v

    # Start with base weights, merge LoRA deltas
    merged = dict(base_weights)
    for key in list(merged.keys()):
        if key in lora_a and key in lora_b:
            a = lora_a[key]
            b = lora_b[key]
            w = merged[key]
            if a.dim() == 2 and b.dim() == 2:
                merged[key] = w + scaling * (b @ a)
            elif a.dim() == 4 and b.dim() == 4:
                # Conv2d LoRA: A=[r,Cin,kH,kW], B=[Cout,r,1,1]
                a_flat = a.reshape(a.shape[0], -1)
                b_flat = b.reshape(b.shape[0], -1)
                delta = (b_flat @ a_flat).reshape(w.shape)
                merged[key] = w + scaling * delta

    # Combine
    merged.update(plain_weights)

    # Load into model
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
# 2. Attention hook: capture attention weights
# ──────────────────────────────────────────────────────────────────────────────

class AttentionHook:
    """
    Register forward hooks on all Attention modules in a timm ViT to capture
    the raw attention weights (before softmax dropout).
    """

    def __init__(self, model):
        self.attentions = []   # list of (B, heads, N, N) tensors
        self._hooks = []

        for module in model.modules():
            if isinstance(module, Attention):
                hook = module.register_forward_hook(self._hook_fn)
                self._hooks.append(hook)

    def _hook_fn(self, module, input, output):
        """
        Recompute attention weights manually (since fused attention path
        in timm doesn't store them).
        """
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        scale = module.head_dim ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        self.attentions.append(attn.detach().cpu())

    def clear(self):
        self.attentions.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Image preprocessing (matching openvla-oft's DINOv2 transform)
# ──────────────────────────────────────────────────────────────────────────────

def get_dino_transform(model, image_size: int = 224):
    """Build the same preprocessing pipeline openvla-oft uses for DINOv2."""
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, image_size, image_size)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    # Override the first Resize to do exact (H, W) resize (resize-naive strategy)
    from torchvision.transforms import Compose, Resize
    assert isinstance(transform, Compose)
    assert isinstance(transform.transforms[0], Resize)
    target_size = (image_size, image_size)
    transform = Compose([
        Resize(target_size, interpolation=transform.transforms[0].interpolation),
        *transform.transforms[1:],
    ])
    return transform


# ──────────────────────────────────────────────────────────────────────────────
# 4. Video frame extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = 10):
    """Extract `num_frames` evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Cannot read video: {video_path}")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((int(idx), Image.fromarray(frame_rgb)))
    cap.release()
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# 5. Attention map visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_attention(
    model,
    transform,
    frames,            # list of (frame_idx, PIL.Image)
    output_dir: str,
    tag: str,          # e.g. "clean_ep0" or "randomized_ep0"
    device: torch.device,
    image_size: int = 224,
    patch_size: int = 14,
):
    """
    For each frame, run DINOv2 forward pass, extract the last-layer attention
    from all heads, and save a heatmap overlay.

    The DINOv2 ViT-L with reg4 has tokens: [CLS, reg0..reg3, patch0..patch_N-1].
    We visualize the CLS token's attention to the patches (excluding register tokens).
    """
    os.makedirs(output_dir, exist_ok=True)

    hook = AttentionHook(model)
    num_patches_per_side = image_size // patch_size  # 224/14 = 16
    num_patch_tokens = num_patches_per_side ** 2     # 256
    # DINOv2 with reg4: tokens = [CLS] + [4 reg tokens] + [256 patches] = 261 total
    num_prefix_tokens = 1 + 4  # CLS + 4 register tokens

    for frame_idx, pil_img in frames:
        hook.clear()

        # Preprocess
        img_tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        with torch.no_grad():
            _ = model.forward_features(img_tensor)

        # Get the LAST layer's attention: (1, num_heads, N, N)
        last_attn = hook.attentions[-1]  # last block
        # Average over heads -> (N, N)
        attn_avg = last_attn[0].mean(dim=0)  # (N, N)

        # CLS token's attention to patch tokens
        # CLS is token 0, patches start at index num_prefix_tokens
        cls_attn_to_patches = attn_avg[0, num_prefix_tokens:num_prefix_tokens + num_patch_tokens]
        # Reshape to 2D grid
        attn_map = cls_attn_to_patches.reshape(num_patches_per_side, num_patches_per_side).numpy()

        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Resize attention map to original image size for overlay
        orig_w, orig_h = pil_img.size
        attn_map_resized = cv2.resize(attn_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Create figure with original image + heatmap overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Original (frame {frame_idx})")
        axes[0].axis("off")

        # Attention heatmap only
        im = axes[1].imshow(attn_map_resized, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("DINOv2 CLS Attention")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(pil_img)
        axes[2].imshow(attn_map_resized, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.suptitle(f"{tag} — Frame {frame_idx}", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{tag}_frame{frame_idx:04d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # Also create a multi-head visualization for the last frame (to show head diversity)
    last_attn = hook.attentions[-1]  # (1, num_heads, N, N)
    num_heads = last_attn.shape[1]
    cols = 4
    rows = (min(num_heads, 16) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for h in range(min(num_heads, 16)):
        head_attn = last_attn[0, h, 0, num_prefix_tokens:num_prefix_tokens + num_patch_tokens]
        head_map = head_attn.reshape(num_patches_per_side, num_patches_per_side).numpy()
        head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
        head_map_resized = cv2.resize(head_map, (pil_img.size[0], pil_img.size[1]), interpolation=cv2.INTER_CUBIC)

        axes[h].imshow(pil_img)
        axes[h].imshow(head_map_resized, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        axes[h].set_title(f"Head {h}", fontsize=10)
        axes[h].axis("off")

    for i in range(min(num_heads, 16), len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{tag} — Per-Head Attention (Last Layer, Last Frame)", fontsize=14)
    plt.tight_layout()
    head_path = os.path.join(output_dir, f"{tag}_multihead_last_frame.png")
    fig.savefig(head_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {head_path}")

    hook.remove()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv2 attention maps from OpenVLA-OFT")
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
        default="/home/qdhe/workspace/dda4210/project/dataset/data_raw/move_can_pot",
        help="Root directory containing aloha-agilex_clean_50 and aloha-agilex_randomized_500",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_attention_vis",
        help="Directory to save visualization outputs",
    )
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to sample per episode")
    parser.add_argument("--clean_episode", type=int, default=0, help="Which episode to use from clean set")
    parser.add_argument("--randomized_episode", type=int, default=0, help="Which episode to use from randomized set")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size for DINOv2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

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

    # ── Build transform ──
    transform = get_dino_transform(model, image_size=args.image_size)

    # ── Process clean episode ──
    clean_video = os.path.join(
        args.data_root, "aloha-agilex_clean_50", "video", f"episode{args.clean_episode}.mp4"
    )
    print(f"\n[INFO] Processing CLEAN episode: {clean_video}")
    clean_frames = extract_frames(clean_video, args.num_frames)
    print(f"  Extracted {len(clean_frames)} frames")
    visualize_attention(
        model, transform, clean_frames,
        output_dir=args.output_dir,
        tag=f"clean_ep{args.clean_episode}",
        device=device,
        image_size=args.image_size,
    )

    # ── Process randomized episode ──
    randomized_video = os.path.join(
        args.data_root, "aloha-agilex_randomized_500", "video", f"episode{args.randomized_episode}.mp4"
    )
    print(f"\n[INFO] Processing RANDOMIZED episode: {randomized_video}")
    rand_frames = extract_frames(randomized_video, args.num_frames)
    print(f"  Extracted {len(rand_frames)} frames")
    visualize_attention(
        model, transform, rand_frames,
        output_dir=args.output_dir,
        tag=f"randomized_ep{args.randomized_episode}",
        device=device,
        image_size=args.image_size,
    )

    print(f"\n[DONE] All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
