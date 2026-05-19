"""
generate_attention_heatmaps.py

对 500 张提取帧执行 TTA (Test-Time Augmentation) + DINOv2 Attention 可视化。

输入:  extracted_1_per_episode/ 下的 500 张 jpg 图片
输出:  每张图 → 1 个文件夹 → 7 个子文件夹:
       original / shift_up / shift_down / shift_left / shift_right / flip_horizontal / averaged
       每个子文件夹包含 combined.jpg  (原图 | 热力图 | overlay 三合一)

算法:
  1. 加载 DINOv2 ViT-L（支持加载 fine-tuned 权重）
  2. 对每张图施加 6 种变换 (原图 + 4方向平移 + 水平翻转)
  3. 批量送入 DINOv2 提取 16×16 patch 特征
  4. 用 L2-norm 将特征降维为 attention map
  5. 逆变换 + 加权平均 → averaged attention
  6. 生成热力图 (inferno colormap) 和 overlay, 横向拼接保存

参考: build_dino_feature_dataset.py (变换/逆变换/DINOv2 加载)
      process_attention_tta_fusion.py (热力图可视化)

Usage:
    python generate_attention_heatmaps.py \
        --checkpoint_dir ./checkpoints/aloha_100000/openvla-7b+aloha_move_can_pot+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--100000_chkpt \
        --input_dir ./extracted_1_per_episode \
        --output_dir ./attention_tta_output \
        --device cuda:0 \
        --batch_size 16
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import timm
import torch
import matplotlib.cm as cm
from PIL import Image
from torchvision.transforms import Compose, Resize
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. DINOv2 模型构建 & 权重加载 (来自 build_dino_feature_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_dinov2(image_size: int = 224):
    """通过 timm 创建 DINOv2-ViT-L, 与 openvla-oft 配置一致."""
    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        img_size=image_size,
    )
    model.eval()
    return model


def load_finetuned_weights(model, checkpoint_path: str):
    """从 openvla-oft 的 vision_backbone checkpoint 加载 fine-tuned DINOv2 权重."""
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


def get_dino_transform(model, image_size: int = 224):
    """构建与 openvla-oft 一致的 DINOv2 预处理 pipeline."""
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
# 2. DINOv2 CLS token attention 提取 (批量)
# ──────────────────────────────────────────────────────────────────────────────

def extract_cls_attention_batch(model, img_batch, device, image_size=224, patch_size=14):
    """
    批量提取 DINOv2 最后一层 CLS token 对 patch tokens 的 attention weights.
    通过 forward hook 捕获 self-attention 层的 attention weights.

    Returns: (B, 16, 16) numpy array (多头平均, 归一化到 [0,1])
    """
    num_patches_per_side = image_size // patch_size  # 16
    num_patch_tokens = num_patches_per_side ** 2      # 256

    # 用 hook 捕获最后一层 attention weights
    attn_weights_store = {}

    def _hook_fn(module, input, output):
        # timm ViT 的 Attention 模块在 forward 中计算 attn = (q @ k.T) * scale
        # 我们需要手动重新计算 attention weights
        pass

    # 直接 monkey-patch 最后一个 block 的 attn 模块来捕获 attention
    last_block = model.blocks[-1]
    original_forward = last_block.attn.forward

    def _patched_forward(x):
        B, N, C = x.shape
        qkv = last_block.attn.qkv(x).reshape(B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
        attn = attn.softmax(dim=-1)
        attn_weights_store['attn'] = attn.detach()  # (B, num_heads, N, N)

        attn = last_block.attn.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = last_block.attn.proj(x)
        x = last_block.attn.proj_drop(x)
        return x

    # Patch, forward, restore
    last_block.attn.forward = _patched_forward
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model(img_batch.to(device))
    finally:
        last_block.attn.forward = original_forward

    # 提取 CLS token 对 patch tokens 的 attention
    attn = attn_weights_store['attn']  # (B, num_heads, N, N)
    # N = 1 (CLS) + 4 (register) + 256 (patches) for reg4 model
    # CLS token 是第 0 个 token
    # Patch tokens 从第 5 个开始 (skip CLS + 4 registers)
    num_prefix_tokens = 1 + model.num_reg_tokens  # CLS + registers
    cls_attn = attn[:, :, 0, num_prefix_tokens:]  # (B, num_heads, 256)
    cls_attn = cls_attn.mean(dim=1)  # 多头平均 → (B, 256)
    cls_attn = cls_attn.float().cpu().numpy()

    B = cls_attn.shape[0]
    attn_maps = []
    for i in range(B):
        a = cls_attn[i].reshape(num_patches_per_side, num_patches_per_side)
        mn, mx = a.min(), a.max()
        if mx - mn > 1e-8:
            a = (a - mn) / (mx - mn)
        else:
            a = np.zeros_like(a)
        attn_maps.append(a.astype(np.float32))

    return np.stack(attn_maps, axis=0)  # (B, 16, 16)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 几何变换 (来自 build_dino_feature_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def apply_shift(pil_img, direction, k, patch_size=14):
    """对 PIL 图像施加像素级平移, 空出部分用白色 (255) 填充."""
    img = np.array(pil_img)
    H, W = img.shape[:2]
    shift_px = k * patch_size
    result = np.full_like(img, 255)

    if direction == 'right':
        result[:, shift_px:, :] = img[:, :W - shift_px, :]
    elif direction == 'left':
        result[:, :W - shift_px, :] = img[:, shift_px:, :]
    elif direction == 'down':
        result[shift_px:, :, :] = img[:H - shift_px, :, :]
    elif direction == 'up':
        result[:H - shift_px, :, :] = img[shift_px:, :, :]
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return Image.fromarray(result)


def apply_hflip(pil_img):
    """水平翻转."""
    return pil_img.transpose(Image.FLIP_LEFT_RIGHT)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 逆变换 (将 attention map 映射回原始坐标)
# ──────────────────────────────────────────────────────────────────────────────

def reverse_align_shift(attn_map, direction, k, grid_size=16):
    """
    逆平移: 将平移后的 attention 映射回原始坐标.
    Returns: (aligned_map, mask)  shape (grid_size, grid_size)
    """
    aligned = np.zeros((grid_size, grid_size), dtype=np.float32)
    mask = np.ones((grid_size, grid_size), dtype=np.float32)

    if direction == 'right':
        aligned[:, :grid_size - k] = attn_map[:, k:]
        mask[:, grid_size - k:] = 0
    elif direction == 'left':
        aligned[:, k:] = attn_map[:, :grid_size - k]
        mask[:, :k] = 0
    elif direction == 'down':
        aligned[:grid_size - k, :] = attn_map[k:, :]
        mask[grid_size - k:, :] = 0
    elif direction == 'up':
        aligned[k:, :] = attn_map[:grid_size - k, :]
        mask[:k, :] = 0
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return aligned, mask


def reverse_align_hflip(attn_map):
    """逆水平翻转: 将翻转后的 attention 映射回原始坐标."""
    aligned = np.flip(attn_map, axis=1).copy()
    mask = np.ones_like(attn_map, dtype=np.float32)
    return aligned, mask


# ──────────────────────────────────────────────────────────────────────────────
# 5. 可视化
# ──────────────────────────────────────────────────────────────────────────────


def create_heatmap(attn_map, target_h, target_w):
    """
    从 16×16 attention map 生成高分辨率热力图 (inferno colormap).
    Returns: RGB (target_h, target_w, 3) uint8
    """
    upsampled = cv2.resize(attn_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    upsampled = np.clip(upsampled, 0, 1)
    cmap = cm.get_cmap('inferno')
    colored = cmap(upsampled)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def create_overlay(image_np, attn_map, alpha=0.4):
    """
    将热力图叠加到原图上.
    Args:
        image_np: RGB (H, W, 3) uint8
        attn_map: (16, 16) float32, [0, 1]
    Returns: RGB (H, W, 3) uint8
    """
    h, w = image_np.shape[:2]
    heatmap = create_heatmap(attn_map, h, w)
    overlay = (alpha * heatmap.astype(np.float32) + (1 - alpha) * image_np.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_combined_image(image_np, attn_map):
    """
    生成三合一拼接图: [原图 | 热力图 | overlay]
    Args:
        image_np: RGB (H, W, 3) uint8
        attn_map: (16, 16) float32, [0, 1]
    Returns: RGB (H, W*3, 3) uint8
    """
    h, w = image_np.shape[:2]
    heatmap = create_heatmap(attn_map, h, w)
    overlay = create_overlay(image_np, attn_map)
    combined = np.concatenate([image_np, heatmap, overlay], axis=1)
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# 6. 主处理逻辑
# ──────────────────────────────────────────────────────────────────────────────

# 变换定义: (名称, 类型, 参数)
TRANSFORM_SPECS = [
    ("original",        "none",  None),
    ("shift_up",        "shift", "up"),
    ("shift_down",      "shift", "down"),
    ("shift_left",      "shift", "left"),
    ("shift_right",     "shift", "right"),
    ("flip_horizontal", "hflip", None),
]


def process_single_image(model, dino_transform, pil_img_224, device, shift_k,
                          image_size=224, patch_size=14, grid_size=16):
    """
    对单张图处理所有变换, 提取 attention, 逆变换, 求平均.

    Returns:
        results: dict  name -> {
            "image_np": (224, 224, 3) uint8,
            "attn_map": (16, 16) float32
        }
        其中包含 6 个变换 + 1 个 "averaged"
    """
    # -- 准备所有变换图像 --
    transformed_pils = []
    for name, ttype, param in TRANSFORM_SPECS:
        if ttype == "none":
            transformed_pils.append(pil_img_224.copy())
        elif ttype == "shift":
            transformed_pils.append(apply_shift(pil_img_224, param, shift_k, patch_size))
        elif ttype == "hflip":
            transformed_pils.append(apply_hflip(pil_img_224))

    # -- 批量提取 CLS token attention --
    batch_tensors = torch.stack([dino_transform(img) for img in transformed_pils], dim=0)
    all_attn_maps = extract_cls_attention_batch(
        model, batch_tensors, device, image_size, patch_size
    )  # (6, 16, 16) numpy

    # -- 逐变换: 逆变换用于求 average --
    results = {}
    Q = np.zeros((grid_size, grid_size), dtype=np.float32)
    K = np.zeros((grid_size, grid_size), dtype=np.float32)

    for idx, (name, ttype, param) in enumerate(TRANSFORM_SPECS):
        attn_map = all_attn_maps[idx]  # (16, 16) already normalized

        results[name] = {
            "image_np": np.array(transformed_pils[idx]),
            "attn_map": attn_map,
        }

        # 逆变换到原始坐标, 累加用于求平均
        if ttype == "none":
            aligned, mask = attn_map.copy(), np.ones_like(attn_map)
        elif ttype == "shift":
            aligned, mask = reverse_align_shift(attn_map, param, shift_k, grid_size)
        elif ttype == "hflip":
            aligned, mask = reverse_align_hflip(attn_map)

        Q += aligned * mask
        K += mask

    # -- Averaged --
    K_safe = np.maximum(K, 1.0)
    avg_attn = Q / K_safe
    # 重新归一化
    mn, mx = avg_attn.min(), avg_attn.max()
    if mx - mn > 1e-8:
        avg_attn = (avg_attn - mn) / (mx - mn)

    results["averaged"] = {
        "image_np": np.array(pil_img_224),  # averaged 用原图
        "attn_map": avg_attn,
    }

    return results


def save_results(results, output_subdir: Path):
    """保存所有 7 个变换的结果到子文件夹."""
    folder_order = [
        "original", "shift_up", "shift_down",
        "shift_left", "shift_right", "flip_horizontal", "averaged"
    ]
    for name in folder_order:
        data = results[name]
        folder = output_subdir / name
        folder.mkdir(parents=True, exist_ok=True)

        image_np = data["image_np"]
        attn_map = data["attn_map"]

        # 保存三合一拼接图
        combined = create_combined_image(image_np, attn_map)
        Image.fromarray(combined).save(folder / "combined.jpg", quality=95)

        # 同时保存单独文件 (方便查看)
        Image.fromarray(image_np).save(folder / "image.jpg", quality=95)
        heatmap = create_heatmap(attn_map, image_np.shape[0], image_np.shape[1])
        Image.fromarray(heatmap).save(folder / "attention_heatmap.jpg", quality=95)
        overlay = create_overlay(image_np, attn_map)
        Image.fromarray(overlay).save(folder / "overlay.jpg", quality=95)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 TTA Attention 热力图生成"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="/data1/jiangshaohan/WuYuhao/checkpoints/aloha_100000/"
                "openvla-7b+aloha_move_can_pot+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--100000_chkpt",
        help="openvla-oft checkpoint 目录 (含 vision_backbone--*_checkpoint.pt)",
    )
    parser.add_argument(
        "--input_dir", type=str,
        default="/data1/jiangshaohan/WuYuhao/extracted_1_per_episode",
        help="输入图片目录",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/data1/jiangshaohan/WuYuhao/attention_tta_output",
        help="输出目录",
    )
    parser.add_argument("--shift_k", type=int, default=1,
                        help="平移 patch 数 (default: 1 = 14px)")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每次处理多少张图 (每张图内部 6 个变换已批处理)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patch_size = 14
    grid_size = args.image_size // patch_size

    print("=" * 70)
    print("DINOv2 TTA Attention 热力图生成")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"输入: {args.input_dir}")
    print(f"输出: {args.output_dir}")
    print(f"平移: {args.shift_k} patch = {args.shift_k * patch_size} px")
    print(f"变换: original + 上下左右平移 + 水平翻转 + averaged = 7 folders")
    print()

    # ── 构建 DINOv2 ──
    print("[1/3] 构建 DINOv2 ViT-L ...")
    model = build_dinov2(image_size=args.image_size)

    # ── 加载 fine-tuned 权重 ──
    vision_ckpt_pattern = "vision_backbone--*_checkpoint.pt"
    ckpt_dir = Path(args.checkpoint_dir)
    vision_ckpts = sorted(ckpt_dir.glob("vision_backbone*checkpoint.pt"))
    if vision_ckpts:
        vision_ckpt = str(vision_ckpts[0])
        print(f"[2/3] 加载 fine-tuned 权重: {vision_ckpt}")
        model = load_finetuned_weights(model, vision_ckpt)
    else:
        print(f"[2/3] 未找到 fine-tuned 权重, 使用 pretrained 权重")

    model = model.to(device)
    model.eval()

    dino_transform = get_dino_transform(model, image_size=args.image_size)

    # ── 扫描输入图片 ──
    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(input_dir.glob("*.png"))
    print(f"[3/3] 找到 {len(image_files)} 张图片")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 处理 ──
    successful = 0
    failed = 0

    pbar = tqdm(image_files, desc="处理进度", unit="img")
    for image_path in pbar:
        stem = image_path.stem
        output_subdir = output_dir / stem

        # 跳过已处理的
        if (output_subdir / "averaged" / "combined.jpg").exists():
            successful += 1
            pbar.set_postfix(ok=successful, fail=failed)
            continue

        try:
            # 加载并 resize
            pil_img = Image.open(image_path).convert("RGB")
            pil_img_224 = pil_img.resize((args.image_size, args.image_size), Image.BILINEAR)

            # 处理
            results = process_single_image(
                model, dino_transform, pil_img_224, device,
                shift_k=args.shift_k,
                image_size=args.image_size,
                patch_size=patch_size,
                grid_size=grid_size,
            )

            # 保存
            save_results(results, output_subdir)
            successful += 1

        except Exception as e:
            print(f"\n[ERROR] {stem}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        pbar.set_postfix(ok=successful, fail=failed)

    # ── Summary ──
    print()
    print("=" * 70)
    print(f"Completed! Success: {successful}/{len(image_files)}, Failed: {failed}")
    print(f"Output Directory: {output_dir}")
    print()
    print("Output Structure:")
    print(f"  {output_dir.name}/")
    print(f"  └── episodeX_frame_YYYYYY/")
    print(f"      ├── original/")
    print(f"      │   ├── combined.jpg      (Original | Heatmap | Overlay)")
    print(f"      │   ├── image.jpg")
    print(f"      │   ├── attention_heatmap.jpg")
    print(f"      │   └── overlay.jpg")
    print(f"      ├── shift_up/    ...")
    print(f"      ├── shift_down/  ...")
    print(f"      ├── shift_left/  ...")
    print(f"      ├── shift_right/ ...")
    print(f"      ├── flip_horizontal/ ...")
    print(f"      └── averaged/    ...")
    print("=" * 70)


if __name__ == "__main__":
    main()
