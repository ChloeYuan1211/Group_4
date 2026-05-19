import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import to_pil_image


EPISODE_RE = re.compile(r'episode(\d+)$')
FRAME_RE = re.compile(r'frame_(\d+)\.pt$')


def parse_episode_index(name: str) -> int:
    match = EPISODE_RE.match(name)
    if not match:
        raise ValueError(f'Unexpected episode name: {name}')
    return int(match.group(1))


def parse_frame_index(name: str) -> int:
    match = FRAME_RE.match(name)
    if not match:
        raise ValueError(f'Unexpected frame name: {name}')
    return int(match.group(1))


@dataclass(frozen=True)
class FrameRecord:
    domain: str
    root: str
    episode: str
    frame: int
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain': self.domain,
            'root': self.root,
            'episode': self.episode,
            'frame': self.frame,
            'path': self.path,
        }


def _scan_domain_root(root: str, domain: str) -> Dict[str, List[FrameRecord]]:
    root_path = Path(root)
    episodes: Dict[str, List[FrameRecord]] = {}
    for episode_dir in sorted(root_path.glob('episode*'), key=lambda p: parse_episode_index(p.name)):
        frame_records: List[FrameRecord] = []
        for frame_path in sorted(episode_dir.glob('frame_*.pt'), key=lambda p: parse_frame_index(p.name)):
            frame_records.append(
                FrameRecord(
                    domain=domain,
                    root=str(root_path),
                    episode=episode_dir.name,
                    frame=parse_frame_index(frame_path.name),
                    path=str(frame_path),
                )
            )
        if frame_records:
            episodes[episode_dir.name] = frame_records
    if not episodes:
        raise FileNotFoundError(f'No frame_*.pt samples found under {root_path}')
    return episodes


def _split_episode_names(episode_names: Sequence[str], train_count: int, seed: int) -> tuple[List[str], List[str]]:
    shuffled = list(episode_names)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_names = sorted(shuffled[:train_count], key=parse_episode_index)
    val_names = sorted(shuffled[train_count:], key=parse_episode_index)
    return train_names, val_names


class DINOFeatureRepository:
    def __init__(self, episodes_by_domain: Dict[str, Dict[str, List[FrameRecord]]], split_manifest: Dict[str, Any]) -> None:
        self.episodes_by_domain = episodes_by_domain
        self.split_manifest = split_manifest

    @classmethod
    def from_roots(cls, randomized_root: str, clean_root: str, seed: int = 42) -> 'DINOFeatureRepository':
        randomized = _scan_domain_root(randomized_root, 'randomized')
        clean = _scan_domain_root(clean_root, 'clean')

        train_randomized, val_randomized = _split_episode_names(sorted(randomized.keys(), key=parse_episode_index), 450, seed)
        train_clean, val_clean = _split_episode_names(sorted(clean.keys(), key=parse_episode_index), 40, seed)

        split_manifest = {
            'seed': seed,
            'roots': {'randomized': randomized_root, 'clean': clean_root},
            'splits': {
                'train_randomized': train_randomized,
                'val_randomized': val_randomized,
                'train_clean': train_clean,
                'val_clean': val_clean,
            },
        }
        return cls({'randomized': randomized, 'clean': clean}, split_manifest)

    def estimate_train_samples_per_epoch(self, randomized_stride: int = 4, clean_stride: int = 2) -> int:
        total = 0.0
        for episode in self.split_manifest['splits']['train_randomized']:
            frames = self.episodes_by_domain['randomized'][episode]
            total += sum(len(frames[offset::randomized_stride]) for offset in range(randomized_stride)) / randomized_stride
        for episode in self.split_manifest['splits']['train_clean']:
            frames = self.episodes_by_domain['clean'][episode]
            total += sum(len(frames[offset::clean_stride]) for offset in range(clean_stride)) / clean_stride
        return int(round(total))

    def build_train_epoch_state(
        self,
        epoch: int,
        randomized_stride: int = 4,
        clean_stride: int = 2,
        seed: int = 42,
    ) -> Dict[str, Any]:
        rng = random.Random(seed + epoch)
        offsets = {'randomized': {}, 'clean': {}}
        samples: List[Dict[str, Any]] = []

        for episode in self.split_manifest['splits']['train_randomized']:
            frames = self.episodes_by_domain['randomized'][episode]
            offset = rng.randrange(randomized_stride)
            offsets['randomized'][episode] = offset
            samples.extend(record.to_dict() for record in frames[offset::randomized_stride])

        for episode in self.split_manifest['splits']['train_clean']:
            frames = self.episodes_by_domain['clean'][episode]
            offset = rng.randrange(clean_stride)
            offsets['clean'][episode] = offset
            samples.extend(record.to_dict() for record in frames[offset::clean_stride])

        generator = torch.Generator()
        generator.manual_seed(seed + epoch * 100003 + 17)
        global_indices = torch.randperm(len(samples), generator=generator).tolist() if samples else []

        return {
            'epoch': epoch,
            'train_offsets_randomized': offsets['randomized'],
            'train_offsets_clean': offsets['clean'],
            'samples': samples,
            'global_indices': global_indices,
        }

    def build_val_samples(self, randomized_stride: int = 4, clean_stride: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        val_randomized: List[Dict[str, Any]] = []
        val_clean: List[Dict[str, Any]] = []

        for episode in self.split_manifest['splits']['val_randomized']:
            frames = self.episodes_by_domain['randomized'][episode]
            val_randomized.extend(record.to_dict() for record in frames[0::randomized_stride])

        for episode in self.split_manifest['splits']['val_clean']:
            frames = self.episodes_by_domain['clean'][episode]
            val_clean.extend(record.to_dict() for record in frames[0::clean_stride])

        return {'randomized': val_randomized, 'clean': val_clean}


class CombinedFrameDataset(Dataset):
    def __init__(self, samples: Sequence[Mapping[str, Any]], image_transform) -> None:
        self.samples = list(samples)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.samples[idx])
        data = torch.load(sample['path'], map_location='cpu')
        image_tensor = data['image'].clamp(0.0, 1.0)
        pil_image = to_pil_image(image_tensor)
        pixel_values = self.image_transform(pil_image)
        feature_gt = data['feature_gt'].reshape(-1, data['feature_gt'].shape[-1]).float()
        valid_count = data['valid_count'].float()
        return {
            'pixel_values': pixel_values,
            'feature_gt': feature_gt,
            'valid_count': valid_count,
            'episode': data['episode'],
            'frame': data['frame'],
            'domain': sample['domain'],
            'path': sample['path'],
        }


class OrderedIndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
