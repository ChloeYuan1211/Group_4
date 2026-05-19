from typing import Dict

import torch


def _flatten_patch_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return features.reshape(features.shape[0], -1, features.shape[-1])
    if features.ndim == 3:
        return features
    raise ValueError(f"Expected feature tensor with ndim 3 or 4, got shape {tuple(features.shape)}")


def _flatten_valid_count(valid_count: torch.Tensor | None, num_patches: int) -> torch.Tensor | None:
    if valid_count is None:
        return None
    if valid_count.ndim == 3:
        flattened = valid_count.reshape(valid_count.shape[0], -1)
    elif valid_count.ndim == 2:
        flattened = valid_count
    else:
        raise ValueError(f"Expected valid_count with ndim 2 or 3, got shape {tuple(valid_count.shape)}")
    if flattened.shape[1] != num_patches:
        raise ValueError(f"valid_count patch dimension mismatch: expected {num_patches}, got {flattened.shape[1]}")
    return flattened.float()


def feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    valid_count: torch.Tensor | None = None,
    loss_cos_weight: float = 1.0,
    loss_mse_weight: float = 1.0,
    use_valid_count_weight: bool = True,
    max_valid_count: float = 5.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    student = _flatten_patch_features(student_features).float()
    teacher = _flatten_patch_features(teacher_features).float()
    if student.shape != teacher.shape:
        raise ValueError(f"Student / teacher shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}")

    flat_valid_count = _flatten_valid_count(valid_count, student.shape[1])
    if use_valid_count_weight and flat_valid_count is not None:
        patch_weight = flat_valid_count / max_valid_count
    else:
        patch_weight = torch.ones(student.shape[:2], device=student.device, dtype=student.dtype)

    cosine_numerator = (teacher * student).sum(dim=-1)
    cosine_denominator = teacher.norm(dim=-1) * student.norm(dim=-1) + eps
    patch_cos = 1.0 - (cosine_numerator / cosine_denominator)
    patch_mse = ((teacher - student) ** 2).mean(dim=-1)

    weight_denom = patch_weight.sum().clamp_min(eps)
    loss_cos = (patch_weight * patch_cos).sum() / weight_denom
    loss_mse = (patch_weight * patch_mse).sum() / weight_denom
    loss_total = loss_cos_weight * loss_cos + loss_mse_weight * loss_mse

    mean_valid_count = flat_valid_count.mean() if flat_valid_count is not None else torch.tensor(max_valid_count, device=student.device)
    teacher_feature_norm = teacher.norm(dim=-1).mean()
    student_feature_norm = student.norm(dim=-1).mean()

    return {
        "loss_total": loss_total,
        "loss_cos": loss_cos,
        "loss_mse": loss_mse,
        "mean_valid_count": mean_valid_count,
        "teacher_feature_norm": teacher_feature_norm,
        "student_feature_norm": student_feature_norm,
    }
