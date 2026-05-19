import math
from typing import Any, Dict, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize


DEFAULT_BLOCKS_BY_MODE = {
    "full_last4": 4,
    "full_last6": 6,
    "lora_last6": 6,
}


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.dropout_p = dropout

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        dropped = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p > 0 else x
        lora_hidden = F.linear(dropped, self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return result + self.scaling * lora_out

    def merged_weight(self) -> torch.Tensor:
        delta = torch.matmul(self.lora_B, self.lora_A)
        return self.base_layer.weight + self.scaling * delta


def build_dinov2(image_size: int = 224) -> nn.Module:
    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=image_size,
    )
    model.eval()
    return model


def get_dino_transform(model: nn.Module, image_size: int = 224):
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, image_size, image_size)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    assert isinstance(transform, Compose)
    assert isinstance(transform.transforms[0], Resize)
    target_size = (image_size, image_size)
    return Compose([
        Resize(target_size, interpolation=transform.transforms[0].interpolation),
        *transform.transforms[1:],
    ])


def merge_openvla_dino_checkpoint(checkpoint_path: str, lora_alpha: int = 16, lora_r: int = 32) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    prefix = "vision_backbone.featurizer."
    dino_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    scaling = lora_alpha / lora_r
    base_weights: Dict[str, torch.Tensor] = {}
    lora_a: Dict[str, torch.Tensor] = {}
    lora_b: Dict[str, torch.Tensor] = {}
    plain_weights: Dict[str, torch.Tensor] = {}

    for key, value in dino_state.items():
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[2] in ("scale", "shift"):
            continue
        if ".base_layer." in key:
            mapped = key.replace(".base_layer.", ".").replace(".block.", ".")
            base_weights[mapped] = value
        elif ".lora_A.default." in key:
            mapped = key.split(".lora_A.default.")[0].replace(".block.", ".")
            suffix = key.split(".lora_A.default.")[1]
            lora_a[f"{mapped}.{suffix}"] = value
        elif ".lora_B.default." in key:
            mapped = key.split(".lora_B.default.")[0].replace(".block.", ".")
            suffix = key.split(".lora_B.default.")[1]
            lora_b[f"{mapped}.{suffix}"] = value
        else:
            mapped = key.replace(".block.", ".").replace(".scale_factor", ".gamma")
            plain_weights[mapped] = value

    merged = dict(base_weights)
    for key in list(base_weights.keys()):
        if key in lora_a and key in lora_b:
            if lora_a[key].dim() == 2 and lora_b[key].dim() == 2:
                delta = lora_b[key] @ lora_a[key]
            elif lora_a[key].dim() == 4 and lora_b[key].dim() == 4:
                a_flat = lora_a[key].reshape(lora_a[key].shape[0], -1)
                b_flat = lora_b[key].reshape(lora_b[key].shape[0], -1)
                delta = (b_flat @ a_flat).reshape(base_weights[key].shape)
            else:
                raise ValueError(f"Unsupported LoRA tensor ranks for {key}: {lora_a[key].shape}, {lora_b[key].shape}")
            merged[key] = base_weights[key] + scaling * delta
    merged.update(plain_weights)
    return merged


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, leaf_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, leaf_name, new_module)


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _block_index_from_module_name(module_name: str) -> int | None:
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
        return int(parts[1])
    return None


def _resolve_last_n_blocks(backbone: nn.Module, finetune_mode: str, unfreeze_last_n_blocks: int | None) -> Dict[str, Any]:
    if finetune_mode not in DEFAULT_BLOCKS_BY_MODE:
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")

    target_layer_idx = len(backbone.blocks) - 2
    requested = DEFAULT_BLOCKS_BY_MODE[finetune_mode] if unfreeze_last_n_blocks is None else unfreeze_last_n_blocks
    if requested <= 0:
        raise ValueError(f"unfreeze_last_n_blocks must be positive, got {requested}")

    first_block_idx = max(0, target_layer_idx - requested + 1)
    trainable_block_indices = list(range(first_block_idx, target_layer_idx + 1))
    return {
        "target_layer_idx": target_layer_idx,
        "resolved_unfreeze_last_n_blocks": len(trainable_block_indices),
        "trainable_block_indices": trainable_block_indices,
    }


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    r: int,
    alpha: int,
    dropout: float,
    block_indices: Sequence[int],
) -> None:
    block_index_set = set(block_indices)
    replacements = []
    for module_name, module in model.named_modules():
        if not any(module_name.endswith(target) for target in target_modules):
            continue
        block_idx = _block_index_from_module_name(module_name)
        if block_idx is None or block_idx not in block_index_set:
            continue
        if isinstance(module, nn.Linear):
            replacements.append((module_name, module))

    for module_name, module in replacements:
        _replace_module(model, module_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))


def _count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _count_total_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def _collect_trainable_parameter_names(module: nn.Module) -> list[str]:
    return [name for name, param in module.named_parameters() if param.requires_grad]


class DINOStudent(nn.Module):
    def __init__(self, backbone: nn.Module, image_size: int = 224, target_layer_idx: int | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.target_layer_idx = len(self.backbone.blocks) - 2 if target_layer_idx is None else target_layer_idx

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone.get_intermediate_layers(pixel_values, n={self.target_layer_idx})
        return features[0].float()


def build_student_from_openvla_checkpoint(
    checkpoint_path: str,
    image_size: int = 224,
    finetune_mode: str = "full_last4",
    unfreeze_last_n_blocks: int | None = None,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    train_final_norm: bool = True,
) -> tuple[DINOStudent, object, Dict[str, Any]]:
    backbone = build_dinov2(image_size=image_size)
    merged_state = merge_openvla_dino_checkpoint(checkpoint_path, lora_alpha=lora_alpha, lora_r=lora_rank)
    missing, unexpected = backbone.load_state_dict(merged_state, strict=False)
    if missing:
        print(f"[WARN] Missing DINO keys: {missing[:5]} ... ({len(missing)} total)")
    if unexpected:
        print(f"[WARN] Unexpected DINO keys: {unexpected[:5]} ... ({len(unexpected)} total)")

    freeze_module(backbone)

    plan = _resolve_last_n_blocks(backbone, finetune_mode, unfreeze_last_n_blocks)
    trainable_block_indices = plan["trainable_block_indices"]
    if finetune_mode.startswith("full_"):
        for block_idx in trainable_block_indices:
            _unfreeze_module(backbone.blocks[block_idx])
    elif finetune_mode.startswith("lora_"):
        inject_lora(
            backbone,
            ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
            lora_rank,
            lora_alpha,
            lora_dropout,
            trainable_block_indices,
        )
    else:
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")

    if train_final_norm and hasattr(backbone, "norm"):
        _unfreeze_module(backbone.norm)

    student = DINOStudent(backbone=backbone, image_size=image_size, target_layer_idx=plan["target_layer_idx"])
    transform = get_dino_transform(backbone, image_size=image_size)
    finetune_summary = {
        "finetune_mode": finetune_mode,
        "target_layer_idx": plan["target_layer_idx"],
        "student_output_is_prenorm": True,
        "resolved_unfreeze_last_n_blocks": plan["resolved_unfreeze_last_n_blocks"],
        "trainable_blocks": trainable_block_indices,
        "train_final_norm": bool(train_final_norm and hasattr(backbone, "norm")),
        "lora_rank": lora_rank if finetune_mode.startswith("lora_") else None,
        "lora_alpha": lora_alpha if finetune_mode.startswith("lora_") else None,
        "lora_dropout": lora_dropout if finetune_mode.startswith("lora_") else None,
        "trainable_param_count": _count_trainable_parameters(student),
        "total_param_count": _count_total_parameters(student),
        "trainable_param_count_m": round(_count_trainable_parameters(student) / 1_000_000.0, 3),
        "trainable_parameter_names": _collect_trainable_parameter_names(student),
    }
    return student, transform, finetune_summary


def merged_timm_state_dict(student: DINOStudent) -> Dict[str, torch.Tensor]:
    backbone = student.backbone
    reference_state = build_dinov2(image_size=student.image_size).state_dict()
    wrapped_state = backbone.state_dict()
    merged: Dict[str, torch.Tensor] = {}

    for key in reference_state.keys():
        base_weight_key = key.replace(".weight", ".base_layer.weight")
        base_bias_key = key.replace(".bias", ".base_layer.bias")
        if key.endswith(".weight") and base_weight_key in wrapped_state:
            module_name = key.rsplit(".", 1)[0]
            module = backbone.get_submodule(module_name)
            if not isinstance(module, LoRALinear):
                raise TypeError(f"Expected LoRALinear at {module_name}, got {type(module)}")
            merged[key] = module.merged_weight().detach().cpu()
        elif key.endswith(".bias") and base_bias_key in wrapped_state:
            merged[key] = wrapped_state[base_bias_key].detach().cpu()
        elif key in wrapped_state:
            merged[key] = wrapped_state[key].detach().cpu()
        else:
            raise KeyError(f"Unable to reconstruct plain timm key: {key}")

    validator = build_dinov2(image_size=student.image_size)
    validator.load_state_dict(merged, strict=True)
    return merged
