"""
generate_attention_original_only.py

从输入图片目录中逐张读取图片，使用 fine-tuned DINOv2 vision backbone
提取最后一层 CLS token 对 patch tokens 的 attention，并生成 inferno 热力图。

输出目录结构参考 attention_tta_output，但每张图仅保留 original/ 子目录：

output_final/
  episodeX_frame_YYYYYY/
    original/
      image.jpg
      attention_heatmap.jpg
      overlay.jpg
      combined.jpg
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import timm
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize
from tqdm import tqdm


def build_dinov2(image_size: int = 224):
    """Create the DINOv2 ViT-L model used by the OpenVLA vision backbone."""
    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=image_size,
    )
    model.eval()
    return model


def load_finetuned_weights(model, checkpoint_path: str):
    """Load the fine-tuned vision backbone checkpoint and merge LoRA weights."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    prefix = "vision_backbone.featurizer."
    dino_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            dino_state[key[len(prefix):]] = value

    lora_alpha = 16
    lora_r = 32
    scaling = lora_alpha / lora_r

    base_weights = {}
    lora_a = {}
    lora_b = {}
    plain_weights = {}

    for key, value in dino_state.items():
        parts = key.split(".")
        if len(parts) >= 2 and parts[1] in ("scale", "shift"):
            continue

        if ".base_layer." in key:
            param_path = key.replace(".base_layer.", ".").replace(".block.", ".")
            base_weights[param_path] = value
        elif ".lora_A.default." in key:
            param_path = key.split(".lora_A.default.")[0].replace(".block.", ".")
            suffix = key.split(".lora_A.default.")[1]
            lora_a[f"{param_path}.{suffix}"] = value
        elif ".lora_B.default." in key:
            param_path = key.split(".lora_B.default.")[0].replace(".block.", ".")
            suffix = key.split(".lora_B.default.")[1]
            lora_b[f"{param_path}.{suffix}"] = value
        else:
            mapped_key = key.replace(".block.", ".")
            mapped_key = mapped_key.replace(".scale_factor", ".gamma")
            plain_weights[mapped_key] = value

    merged = dict(base_weights)
    for key in list(merged.keys()):
        if key not in lora_a or key not in lora_b:
            continue
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
    return model


def get_dino_transform(model, image_size: int = 224):
    """Match the DINOv2 preprocessing used by the project."""
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, image_size, image_size)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    assert isinstance(transform, Compose)
    assert isinstance(transform.transforms[0], Resize)

    return Compose([
        Resize((image_size, image_size), interpolation=transform.transforms[0].interpolation),
        *transform.transforms[1:],
    ])


def extract_cls_attention(model, img_tensor, device, image_size=224, patch_size=14):
    """
    Extract the last-layer CLS-to-patch attention map and normalize to [0, 1].
    """
    num_patches_per_side = image_size // patch_size
    attn_weights_store = {}

    last_block = model.blocks[-1]
    original_forward = last_block.attn.forward

    def _patched_forward(x):
        batch_size, num_tokens, channels = x.shape
        qkv = last_block.attn.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            last_block.attn.num_heads,
            channels // last_block.attn.num_heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
        attn = attn.softmax(dim=-1)
        attn_weights_store["attn"] = attn.detach()

        attn = last_block.attn.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = last_block.attn.proj(x)
        x = last_block.attn.proj_drop(x)
        return x

    last_block.attn.forward = _patched_forward
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model(img_tensor.to(device))
    finally:
        last_block.attn.forward = original_forward

    attn = attn_weights_store["attn"]
    num_prefix_tokens = 1 + model.num_reg_tokens
    cls_attn = attn[:, :, 0, num_prefix_tokens:]
    cls_attn = cls_attn.mean(dim=1).float().cpu().numpy()[0]

    attn_map = cls_attn.reshape(num_patches_per_side, num_patches_per_side)
    min_value = attn_map.min()
    max_value = attn_map.max()
    if max_value - min_value > 1e-8:
        attn_map = (attn_map - min_value) / (max_value - min_value)
    else:
        attn_map = np.zeros_like(attn_map)
    return attn_map.astype(np.float32)


def create_heatmap(attn_map, target_h, target_w):
    """Upsample the 16x16 attention map and colorize it with inferno."""
    upsampled = cv2.resize(attn_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    upsampled = np.clip(upsampled, 0, 1)
    heatmap = cm.get_cmap("inferno")(upsampled)[:, :, :3]
    return (heatmap * 255).astype(np.uint8)


def create_overlay(image_np, attn_map, alpha=0.4):
    """Blend the inferno heatmap with the RGB image."""
    height, width = image_np.shape[:2]
    heatmap = create_heatmap(attn_map, height, width)
    overlay = alpha * heatmap.astype(np.float32) + (1 - alpha) * image_np.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_combined_image(image_np, attn_map):
    """Concatenate [image | heatmap | overlay] horizontally."""
    height, width = image_np.shape[:2]
    heatmap = create_heatmap(attn_map, height, width)
    overlay = create_overlay(image_np, attn_map)
    return np.concatenate([image_np, heatmap, overlay], axis=1)


def save_outputs(image_np, attn_map, output_dir: Path):
    """Save the image, heatmap, overlay, and combined panel."""
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap = create_heatmap(attn_map, image_np.shape[0], image_np.shape[1])
    overlay = create_overlay(image_np, attn_map)
    combined = create_combined_image(image_np, attn_map)

    Image.fromarray(image_np).save(output_dir / "image.jpg", quality=95)
    Image.fromarray(heatmap).save(output_dir / "attention_heatmap.jpg", quality=95)
    Image.fromarray(overlay).save(output_dir / "overlay.jpg", quality=95)
    Image.fromarray(combined).save(output_dir / "combined.jpg", quality=95)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate original-only CLS attention heatmaps with inferno colormap."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=(
            "/data1/jiangshaohan/WuYuhao/checkpoints/"
            "openvla-7b+aloha_move_can_pot+b4+lr-0.0005+lora-r32+dropout-0.0"
            "--image_aug--100000_chkpt__dinoft_step_0001000"
        ),
        help="Checkpoint directory containing vision_backbone checkpoint.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data1/jiangshaohan/WuYuhao/extracted_1_per_episode",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/jiangshaohan/WuYuhao/output_final",
        help="Directory to save output folders.",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Original-only CLS Attention Heatmap Generation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    model = build_dinov2(image_size=args.image_size)
    ckpt_dir = Path(args.checkpoint_dir)
    vision_ckpts = sorted(ckpt_dir.glob("vision_backbone*checkpoint.pt"))
    if not vision_ckpts:
        raise FileNotFoundError(
            f"No vision backbone checkpoint found in {args.checkpoint_dir}"
        )

    vision_ckpt = str(vision_ckpts[0])
    print(f"Loading fine-tuned vision weights: {vision_ckpt}")
    model = load_finetuned_weights(model, vision_ckpt).to(device)
    model.eval()

    dino_transform = get_dino_transform(model, image_size=args.image_size)

    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if args.limit is not None:
        image_files = image_files[:args.limit]
    if not image_files:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    for image_path in tqdm(image_files, desc="Processing", unit="img"):
        stem = image_path.stem
        target_dir = output_root / stem / "original"
        if (target_dir / "combined.jpg").exists():
            success_count += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_224 = image.resize((args.image_size, args.image_size), Image.BILINEAR)
            image_np = np.array(image_224)
            img_tensor = dino_transform(image_224).unsqueeze(0)

            attn_map = extract_cls_attention(
                model,
                img_tensor,
                device=device,
                image_size=args.image_size,
                patch_size=args.patch_size,
            )
            save_outputs(image_np, attn_map, target_dir)
            success_count += 1
        except Exception as exc:
            fail_count += 1
            print(f"[ERROR] {stem}: {exc}")

    print("=" * 70)
    print(f"Done. success={success_count}, failed={fail_count}")
    print(f"Output directory: {output_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
