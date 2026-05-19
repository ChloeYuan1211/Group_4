import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from modeling import build_student_from_openvla_checkpoint, merged_timm_state_dict
from utils import ensure_dir, save_json


def build_openvla_compatible_checkpoint(
    merged_timm_state: Dict[str, torch.Tensor],
    baseline_vision_backbone: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    baseline_state = torch.load(baseline_vision_backbone, map_location='cpu')
    prefix = 'vision_backbone.featurizer.'
    converted: Dict[str, torch.Tensor] = {}
    replaced_keys = []
    zeroed_lora_keys = []
    preserved_keys = []

    for key, value in baseline_state.items():
        if not key.startswith(prefix):
            converted[key] = value
            preserved_keys.append(key)
            continue

        relative_key = key[len(prefix):]
        parts = relative_key.split('.')
        is_film = len(parts) >= 3 and parts[0] == 'blocks' and parts[2] in ('scale', 'shift')
        if is_film:
            converted[key] = value
            preserved_keys.append(key)
            continue

        if '.lora_A.default.' in relative_key or '.lora_B.default.' in relative_key:
            converted[key] = torch.zeros_like(value)
            zeroed_lora_keys.append(key)
            continue

        plain_key = relative_key.replace('.base_layer.', '.').replace('.block.', '.').replace('.scale_factor', '.gamma')
        if plain_key in merged_timm_state:
            converted[key] = merged_timm_state[plain_key].to(dtype=value.dtype)
            replaced_keys.append({'template_key': key, 'plain_key': plain_key})
        else:
            converted[key] = value
            preserved_keys.append(key)

    metadata = {
        'baseline_vision_backbone': baseline_vision_backbone,
        'replaced_count': len(replaced_keys),
        'zeroed_lora_count': len(zeroed_lora_keys),
        'preserved_count': len(preserved_keys),
        'replaced_keys': replaced_keys,
        'zeroed_lora_keys': zeroed_lora_keys,
        'preserved_keys': preserved_keys,
    }
    return converted, metadata


def export_step_artifacts(
    model,
    baseline_vision_backbone: str,
    output_dir: str,
    tag: str,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    merged_state = merged_timm_state_dict(model)
    standalone_path = output_dir / 'standalone_merged_dino.pt'
    torch.save(merged_state, standalone_path)

    openvla_state, mapping = build_openvla_compatible_checkpoint(merged_state, baseline_vision_backbone)
    vision_backbone_path = output_dir / f'vision_backbone--{tag}.pt'
    torch.save(openvla_state, vision_backbone_path)

    mapping['tag'] = tag
    mapping_path = output_dir / 'export_metadata.json'
    save_json(mapping, mapping_path)
    return {
        'standalone_merged_dino': str(standalone_path),
        'openvla_vision_backbone': str(vision_backbone_path),
        'export_metadata': str(mapping_path),
        'mapping': mapping,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export standalone DINO weights to OpenVLA-compatible vision backbone.')
    parser.add_argument('--baseline_vision_backbone', required=True)
    parser.add_argument('--student_state', required=True, help='Path to student_state.pt saved by training.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tag', default='exported')
    parser.add_argument('--finetune_mode', choices=('full_last4', 'full_last6', 'lora_last6'), default='full_last4')
    parser.add_argument('--unfreeze_last_n_blocks', type=int, default=None)
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _, finetune_summary = build_student_from_openvla_checkpoint(
        args.baseline_vision_backbone,
        finetune_mode=args.finetune_mode,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    state_dict = torch.load(args.student_state, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    artifacts = export_step_artifacts(model, args.baseline_vision_backbone, args.output_dir, args.tag)
    artifacts['finetune_summary'] = finetune_summary
    print(artifacts)


if __name__ == '__main__':
    main()
