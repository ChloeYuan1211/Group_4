import shutil
from pathlib import Path
from typing import Any, Dict

import torch

from export_openvla import export_step_artifacts
from utils import ensure_dir, load_json, save_json, unwrap_model


def save_checkpoint_bundle(
    bundle_dir: str | Path,
    model,
    optimizer,
    scheduler,
    training_state: Dict[str, Any],
    metrics: Dict[str, Any],
    baseline_vision_backbone: str,
    export_tag: str,
) -> Dict[str, Any]:
    bundle_dir = ensure_dir(bundle_dir)

    model_to_save = unwrap_model(model)
    torch.save(model_to_save.state_dict(), bundle_dir / 'student_state.pt')
    torch.save(optimizer.state_dict(), bundle_dir / 'optimizer_state.pt')
    torch.save(scheduler.state_dict() if scheduler is not None else {}, bundle_dir / 'scheduler_state.pt')
    torch.save(training_state['rng_state'], bundle_dir / 'rng_state.pt')

    training_payload = dict(training_state)
    training_payload.pop('rng_state', None)
    torch.save(training_payload, bundle_dir / 'training_state.pt')

    save_json(training_payload['config'], bundle_dir / 'config.json')
    save_json(training_payload['split_manifest'], bundle_dir / 'split_manifest.json')
    save_json(training_payload['epoch_state'], bundle_dir / 'epoch_state.json')
    save_json(metrics, bundle_dir / 'metrics.json')

    export_info = export_step_artifacts(model_to_save, baseline_vision_backbone, str(bundle_dir), export_tag)
    bundle_summary = {
        'global_step': training_payload['global_step'],
        'epoch': training_payload['epoch'],
        'step_in_epoch': training_payload['step_in_epoch'],
        'best_metric': training_payload['best_metric'],
        'metrics': metrics,
        'artifacts': export_info,
    }
    save_json(bundle_summary, bundle_dir / 'bundle_summary.json')
    return bundle_summary


def load_checkpoint_bundle(bundle_dir: str | Path, map_location: str = 'cpu') -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    return {
        'student_state': torch.load(bundle_dir / 'student_state.pt', map_location=map_location),
        'optimizer_state': torch.load(bundle_dir / 'optimizer_state.pt', map_location=map_location),
        'scheduler_state': torch.load(bundle_dir / 'scheduler_state.pt', map_location=map_location),
        'rng_state': torch.load(bundle_dir / 'rng_state.pt', map_location=map_location),
        'training_state': torch.load(bundle_dir / 'training_state.pt', map_location=map_location),
        'metrics': load_json(bundle_dir / 'metrics.json'),
        'bundle_summary': load_json(bundle_dir / 'bundle_summary.json'),
    }


def refresh_alias_from_checkpoint(source_dir: str | Path, alias_dir: str | Path) -> None:
    source_dir = Path(source_dir)
    alias_dir = Path(alias_dir)
    if alias_dir.exists():
        shutil.rmtree(alias_dir)
    shutil.copytree(source_dir, alias_dir)
