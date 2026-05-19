import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from checkpointing import load_checkpoint_bundle, refresh_alias_from_checkpoint, save_checkpoint_bundle
from dataset import CombinedFrameDataset, DINOFeatureRepository, OrderedIndexSampler
from losses import feature_distillation_loss
from modeling import build_student_from_openvla_checkpoint
from utils import (
    AverageMeter,
    append_jsonl,
    barrier,
    cleanup_distributed,
    ensure_dir,
    format_step_dir,
    get_rng_state,
    init_distributed,
    is_main_process,
    move_batch_to_device,
    save_json,
    seed_worker,
    set_rng_state,
    set_seed,
    unwrap_model,
)


METRIC_KEYS = (
    "loss_total",
    "loss_cos",
    "loss_mse",
    "mean_valid_count",
    "teacher_feature_norm",
    "student_feature_norm",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINO-only finetuning V2 for OpenVLA-OFT move_can_pot.")
    parser.add_argument("--randomized_root", default="/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_feature_dataset")
    parser.add_argument("--clean_root", default="/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_feature_dataset_clean50")
    parser.add_argument("--baseline_vision_backbone", default="/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/runs/train_full_singletask/openvla-7b+aloha_move_can_pot+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--100000_chkpt/vision_backbone--100000_checkpoint.pt")
    parser.add_argument("--output_dir", default="/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_finetune_v2/outputs/default")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--grad_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--save_every_steps", type=int, default=200)
    parser.add_argument("--train_log_every_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--finetune_mode", choices=("full_last4", "full_last6", "lora_last6"), default="full_last4")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=None)
    parser.add_argument("--train_final_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--loss_cos_weight", type=float, default=1.0)
    parser.add_argument("--loss_mse_weight", type=float, default=1.0)
    parser.add_argument("--use_valid_count_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_valid_count", type=float, default=5.0)
    parser.add_argument("--randomized_stride", type=int, default=2)
    parser.add_argument("--clean_stride", type=int, default=1)
    parser.add_argument("--candidate_steps", default="1000,2500,4000")
    return parser


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loader(dataset, indices, batch_size, num_workers, pin_memory, prefetch_factor):
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=OrderedIndexSampler(indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def compute_rank_indices(global_indices: List[int], rank: int, world_size: int) -> List[int]:
    return global_indices[rank::world_size]


def parse_candidate_steps(spec: str) -> List[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"candidate step must be positive, got {value}")
        values.append(value)
    return sorted(set(values))


def init_metric_meters() -> Dict[str, AverageMeter]:
    return {key: AverageMeter() for key in METRIC_KEYS}


def update_metric_meters(meters: Dict[str, AverageMeter], metric_values: Dict[str, torch.Tensor], batch_size: int) -> None:
    for key in METRIC_KEYS:
        meters[key].update(metric_values[key].item(), batch_size)


def reduce_metric_meters(meters: Dict[str, AverageMeter], device: torch.device) -> Dict[str, float]:
    keys = list(METRIC_KEYS)
    count = meters[keys[0]].count if keys else 0
    stats = torch.tensor(
        [meters[key].sum for key in keys] + [float(count)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denom = max(stats[-1].item(), 1.0)
    reduced = {key: stats[idx].item() / denom for idx, key in enumerate(keys)}
    reduced["count"] = int(stats[-1].item())
    return reduced


@torch.no_grad()
def evaluate_split(model, loader, device, args):
    model.eval()
    total_sums = {key: 0.0 for key in METRIC_KEYS}
    total_count = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            student_features = model(batch["pixel_values"])
            loss_info = feature_distillation_loss(
                student_features,
                batch["feature_gt"],
                valid_count=batch["valid_count"],
                loss_cos_weight=args.loss_cos_weight,
                loss_mse_weight=args.loss_mse_weight,
                use_valid_count_weight=args.use_valid_count_weight,
                max_valid_count=args.max_valid_count,
            )
        batch_size = batch["feature_gt"].shape[0]
        for key in METRIC_KEYS:
            total_sums[key] += loss_info[key].item() * batch_size
        total_count += batch_size

    stats = torch.tensor(
        [total_sums[key] for key in METRIC_KEYS] + [total_count],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denom = max(stats[-1].item(), 1.0)
    metrics = {key: stats[idx].item() / denom for idx, key in enumerate(METRIC_KEYS)}
    metrics["count"] = int(stats[-1].item())
    return metrics


@torch.no_grad()
def evaluate(model, repo, image_transform, args, rank, world_size, device):
    val_samples = repo.build_val_samples(args.randomized_stride, args.clean_stride)
    randomized_dataset = CombinedFrameDataset(val_samples["randomized"], image_transform)
    clean_dataset = CombinedFrameDataset(val_samples["clean"], image_transform)

    randomized_indices = list(range(len(randomized_dataset)))[rank::world_size]
    clean_indices = list(range(len(clean_dataset)))[rank::world_size]
    randomized_loader = build_loader(
        randomized_dataset, randomized_indices, args.per_device_batch_size, args.num_workers, args.pin_memory, args.prefetch_factor
    )
    clean_loader = build_loader(
        clean_dataset, clean_indices, args.per_device_batch_size, args.num_workers, args.pin_memory, args.prefetch_factor
    )

    randomized_metrics = evaluate_split(model, randomized_loader, device, args)
    clean_metrics = evaluate_split(model, clean_loader, device, args)
    selection_metric = 0.5 * (randomized_metrics["loss_total"] + clean_metrics["loss_total"])
    return {
        "val_randomized": randomized_metrics,
        "val_clean": clean_metrics,
        "selection_metric": selection_metric,
    }


def serializable_args(args: argparse.Namespace, world_size: int, candidate_steps: List[int], finetune_summary: Dict[str, Any]) -> Dict[str, Any]:
    data = vars(args).copy()
    data["candidate_steps"] = candidate_steps
    data["world_size"] = world_size
    data["effective_batch_size"] = args.per_device_batch_size * args.grad_accumulation_steps * world_size
    data["finetune_summary"] = finetune_summary
    return data


def write_candidate_manifest(output_dir: Path, candidate_steps: List[int]) -> None:
    checkpoints_dir = output_dir / "checkpoints"
    payload: Dict[str, Any] = {
        "candidate_steps_requested": candidate_steps,
        "checkpoints": {},
    }
    for step in candidate_steps:
        step_dir = checkpoints_dir / f"step_{step:07d}"
        if step_dir.exists():
            payload["checkpoints"][f"step_{step:07d}"] = str(step_dir)
    for alias_name in ("best_offline", "last"):
        alias_dir = checkpoints_dir / alias_name
        if alias_dir.exists():
            payload["checkpoints"][alias_name] = str(alias_dir)
    save_json(payload, output_dir / "candidate_checkpoints.json")


def main() -> None:
    args = build_parser().parse_args()
    candidate_steps = parse_candidate_steps(args.candidate_steps)
    dist_info = init_distributed()
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]

    if not torch.cuda.is_available():
        raise RuntimeError("This training script expects CUDA/A100. No GPU is available.")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed + rank)

    output_dir = ensure_dir(args.output_dir)
    if is_main_process():
        ensure_dir(output_dir / "checkpoints")

    repo = DINOFeatureRepository.from_roots(args.randomized_root, args.clean_root, seed=args.seed)
    model, image_transform, finetune_summary = build_student_from_openvla_checkpoint(
        args.baseline_vision_backbone,
        finetune_mode=args.finetune_mode,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_final_norm=args.train_final_norm,
    )
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were enabled. Check finetune_mode settings.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    estimated_samples = repo.estimate_train_samples_per_epoch(args.randomized_stride, args.clean_stride)
    estimated_local_micro_batches = math.ceil(estimated_samples / max(world_size, 1) / args.per_device_batch_size)
    estimated_optimizer_steps_per_epoch = math.ceil(estimated_local_micro_batches / args.grad_accumulation_steps)
    total_optimizer_steps = estimated_optimizer_steps_per_epoch * args.num_epochs
    warmup_steps = math.ceil(total_optimizer_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, warmup_steps, total_optimizer_steps)

    if is_main_process():
        save_json(repo.split_manifest, output_dir / "split_manifest.json")
        save_json(
            serializable_args(args, world_size, candidate_steps, finetune_summary),
            output_dir / "config.json",
        )
        print(
            f"[setup] finetune_mode={finetune_summary['finetune_mode']} "
            f"blocks={finetune_summary['trainable_blocks']} "
            f"target_layer_idx={finetune_summary['target_layer_idx']} "
            f"trainable_params={finetune_summary['trainable_param_count']:,}"
        )
        if finetune_summary["train_final_norm"]:
            print("[note] student distillation target uses pre-norm patch tokens; final norm is kept for compatibility / ablation.")

    global_step = 0
    best_metric = float("inf")
    start_epoch = 0
    start_step_in_epoch = 0
    resume_epoch_state = None

    if args.resume_from:
        resume_payload = load_checkpoint_bundle(args.resume_from, map_location="cpu")
        unwrap_model(model).load_state_dict(resume_payload["student_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        set_rng_state(resume_payload["rng_state"])

        training_state = resume_payload["training_state"]
        global_step = int(training_state["global_step"])
        best_metric = float(training_state["best_metric"])
        start_epoch = int(training_state["epoch"])
        start_step_in_epoch = int(training_state["step_in_epoch"])
        resume_epoch_state = training_state["epoch_state"]

        if start_step_in_epoch >= int(training_state.get("optimizer_steps_in_epoch", 10**9)):
            start_epoch += 1
            start_step_in_epoch = 0
            resume_epoch_state = None

    barrier()
    metrics_log_path = output_dir / "metrics.jsonl"
    train_metrics_log_path = output_dir / "train_metrics.jsonl"

    for epoch in range(start_epoch, args.num_epochs):
        if epoch == start_epoch and resume_epoch_state is not None:
            epoch_state = resume_epoch_state
        else:
            epoch_state = repo.build_train_epoch_state(epoch, args.randomized_stride, args.clean_stride, args.seed)

        train_dataset = CombinedFrameDataset(epoch_state["samples"], image_transform)
        local_indices = compute_rank_indices(epoch_state["global_indices"], rank, world_size)
        train_loader = build_loader(
            train_dataset, local_indices, args.per_device_batch_size, args.num_workers, args.pin_memory, args.prefetch_factor
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_metrics = init_metric_meters()
        window_metrics = init_metric_meters()
        micro_batches_to_skip = start_step_in_epoch * args.grad_accumulation_steps if epoch == start_epoch else 0
        optimizer_steps_in_epoch = start_step_in_epoch if epoch == start_epoch else 0
        micro_batch_counter = 0
        epoch_total_optimizer_steps = math.ceil(len(train_loader) / args.grad_accumulation_steps) if len(train_loader) > 0 else 0
        progress_bar = None
        if is_main_process():
            progress_bar = tqdm(
                total=epoch_total_optimizer_steps,
                initial=start_step_in_epoch if epoch == start_epoch else 0,
                desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                dynamic_ncols=True,
                leave=True,
            )

        for micro_idx, batch in enumerate(train_loader):
            if micro_idx < micro_batches_to_skip:
                continue

            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                student_features = model(batch["pixel_values"])
                loss_info = feature_distillation_loss(
                    student_features,
                    batch["feature_gt"],
                    valid_count=batch["valid_count"],
                    loss_cos_weight=args.loss_cos_weight,
                    loss_mse_weight=args.loss_mse_weight,
                    use_valid_count_weight=args.use_valid_count_weight,
                    max_valid_count=args.max_valid_count,
                )
                scaled_loss = loss_info["loss_total"] / args.grad_accumulation_steps

            scaled_loss.backward()
            batch_size = batch["feature_gt"].shape[0]
            update_metric_meters(running_metrics, loss_info, batch_size)
            update_metric_meters(window_metrics, loss_info, batch_size)
            micro_batch_counter += 1

            should_step = (micro_batch_counter % args.grad_accumulation_steps == 0) or (micro_idx == len(train_loader) - 1)
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            optimizer_steps_in_epoch += 1

            if progress_bar is not None:
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                progress_bar.update(1)
                progress_bar.set_postfix(
                    step=global_step,
                    loss=f"{running_metrics['loss_total'].avg:.4f}",
                    cos=f"{running_metrics['loss_cos'].avg:.4f}",
                    mse=f"{running_metrics['loss_mse'].avg:.4f}",
                    lr=f"{current_lr:.2e}",
                )

            if is_main_process() and global_step % 20 == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={running_metrics['loss_total'].avg:.6f} "
                    f"cos={running_metrics['loss_cos'].avg:.6f} "
                    f"mse={running_metrics['loss_mse'].avg:.6f} "
                    f"mean_valid={running_metrics['mean_valid_count'].avg:.3f}"
                )

            if global_step % args.train_log_every_steps == 0:
                window_stats = reduce_metric_meters(window_metrics, device)
                if is_main_process():
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                    train_record = {
                        "record_type": "train",
                        "global_step": global_step,
                        "epoch": epoch,
                        "step_in_epoch": optimizer_steps_in_epoch,
                        "lr": current_lr,
                        "window_size_steps": args.train_log_every_steps,
                        "train_loss_total": window_stats["loss_total"],
                        "train_loss_cos": window_stats["loss_cos"],
                        "train_loss_mse": window_stats["loss_mse"],
                        "mean_valid_count": window_stats["mean_valid_count"],
                        "teacher_feature_norm": window_stats["teacher_feature_norm"],
                        "student_feature_norm": window_stats["student_feature_norm"],
                        "count": window_stats["count"],
                    }
                    append_jsonl(train_record, train_metrics_log_path)
                window_metrics = init_metric_meters()

            should_eval = global_step % args.eval_every_steps == 0
            should_save = global_step % args.save_every_steps == 0
            if not should_eval and not should_save:
                continue

            metrics = evaluate(model, repo, image_transform, args, rank, world_size, device)
            if is_main_process():
                metric_record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": optimizer_steps_in_epoch,
                    "finetune_mode": finetune_summary["finetune_mode"],
                    "trainable_param_count": finetune_summary["trainable_param_count"],
                    "trainable_blocks": finetune_summary["trainable_blocks"],
                    **metrics,
                }
                append_jsonl(metric_record, metrics_log_path)
                print(
                    f"[eval] step={global_step} selection={metrics['selection_metric']:.6f} "
                    f"randomized={metrics['val_randomized']['loss_total']:.6f} "
                    f"clean={metrics['val_clean']['loss_total']:.6f}"
                )

            barrier()

            if should_save and is_main_process():
                training_state = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": optimizer_steps_in_epoch,
                    "optimizer_steps_in_epoch": optimizer_steps_in_epoch,
                    "grad_accum_progress": 0,
                    "best_metric": best_metric,
                    "config": serializable_args(args, world_size, candidate_steps, finetune_summary),
                    "split_manifest": repo.split_manifest,
                    "epoch_state": epoch_state,
                    "rng_state": get_rng_state(),
                }
                step_dir = format_step_dir(output_dir, global_step)
                save_checkpoint_bundle(
                    bundle_dir=step_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_state=training_state,
                    metrics=metrics,
                    baseline_vision_backbone=args.baseline_vision_backbone,
                    export_tag=f"step_{global_step:07d}",
                )

                if metrics["selection_metric"] < best_metric:
                    best_metric = metrics["selection_metric"]
                    training_state["best_metric"] = best_metric
                    best_dir = output_dir / "checkpoints" / "best_offline"
                    refresh_alias_from_checkpoint(step_dir, best_dir)
                    refresh_alias_from_checkpoint(step_dir, output_dir / "checkpoints" / "best")

            barrier()

        if progress_bar is not None:
            progress_bar.close()

        start_step_in_epoch = 0
        resume_epoch_state = None

    final_metrics = evaluate(model, repo, image_transform, args, rank, world_size, device)
    if is_main_process():
        final_training_state = {
            "global_step": global_step,
            "epoch": args.num_epochs - 1,
            "step_in_epoch": estimated_optimizer_steps_per_epoch,
            "optimizer_steps_in_epoch": estimated_optimizer_steps_per_epoch,
            "grad_accum_progress": 0,
            "best_metric": best_metric,
            "config": serializable_args(args, world_size, candidate_steps, finetune_summary),
            "split_manifest": repo.split_manifest,
            "epoch_state": repo.build_train_epoch_state(args.num_epochs - 1, args.randomized_stride, args.clean_stride, args.seed),
            "rng_state": get_rng_state(),
        }
        last_dir = output_dir / "checkpoints" / "last"
        save_checkpoint_bundle(
            bundle_dir=last_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=final_training_state,
            metrics=final_metrics,
            baseline_vision_backbone=args.baseline_vision_backbone,
            export_tag="last",
        )
        if final_metrics["selection_metric"] < best_metric:
            refresh_alias_from_checkpoint(last_dir, output_dir / "checkpoints" / "best_offline")
            refresh_alias_from_checkpoint(last_dir, output_dir / "checkpoints" / "best")
        write_candidate_manifest(output_dir, candidate_steps)

    barrier()
    if is_main_process():
        print("[done] training complete")
    cleanup_distributed()


if __name__ == "__main__":
    main()
