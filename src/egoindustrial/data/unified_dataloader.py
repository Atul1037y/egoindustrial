"""Unified dataloader with domain mixing for multi-dataset training."""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from egoindustrial.data.base_dataset import BaseEgocentricDataset
from egoindustrial.data.epic_kitchens import EpicKitchensDataset
from egoindustrial.data.assembly101 import Assembly101Dataset
from egoindustrial.data.holoassist import HoloAssistDataset


class DomainAwareSampler(Sampler):
    """Sampler that balances batches across domains/datasets."""

    def __init__(
        self,
        dataset_sizes: list[int],
        batch_size: int,
        domain_probs: list[float] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        total = sum(dataset_sizes)
        self.domain_probs = domain_probs or [s / total for s in dataset_sizes]
        self.cumulative_sizes = torch.cumsum(torch.tensor([0] + dataset_sizes), dim=0)

        # Calculate batches per epoch
        self.batches_per_epoch = total // batch_size
        if not drop_last and total % batch_size != 0:
            self.batches_per_epoch += 1

    def __iter__(self):
        # Generate indices for each dataset
        indices_per_domain = []
        for size in self.dataset_sizes:
            idx = torch.arange(size)
            if self.shuffle:
                idx = idx[torch.randperm(size)]
            indices_per_domain.append(idx)

        domain_pointers = [0] * len(self.dataset_sizes)

        for _ in range(self.batches_per_epoch):
            batch_indices = []
            for _ in range(self.batch_size):
                # Sample domain based on probabilities
                domain = torch.multinomial(
                    torch.tensor(self.domain_probs), 1
                ).item()

                if domain_pointers[domain] >= self.dataset_sizes[domain]:
                    # Reshuffle if exhausted
                    if self.shuffle:
                        indices_per_domain[domain] = indices_per_domain[domain][
                            torch.randperm(self.dataset_sizes[domain])
                        ]
                    domain_pointers[domain] = 0

                local_idx = indices_per_domain[domain][domain_pointers[domain]]
                global_idx = self.cumulative_sizes[domain] + local_idx
                batch_indices.append(global_idx.item())
                domain_pointers[domain] += 1

            yield batch_indices

    def __len__(self):
        return self.batches_per_epoch


class ConcatDatasetWithDomain(Dataset):
    """Concatenated dataset that tracks domain membership."""

    def __init__(self, datasets: list[BaseEgocentricDataset]):
        self.datasets = datasets
        self.cumulative_sizes = [0]
        for d in datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(d))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Find which dataset
        dataset_idx = 0
        while dataset_idx < len(self.datasets) and idx >= self.cumulative_sizes[dataset_idx + 1]:
            dataset_idx += 1

        local_idx = idx - self.cumulative_sizes[dataset_idx]
        sample = self.datasets[dataset_idx][local_idx]
        sample["domain_idx"] = dataset_idx
        sample["domain_name"] = self.datasets[dataset_idx].__class__.__name__
        return sample


def build_datasets(cfg: dict[str, Any]) -> list[BaseEgocentricDataset]:
    """Build dataset instances from Hydra config."""
    datasets = []
    dataset_configs = cfg.get("datasets", {})

    for name, ds_cfg in dataset_configs.items():
        root = ds_cfg.pop("root")
        split = ds_cfg.pop("split", "train")

        if name == "epic_kitchens":
            datasets.append(EpicKitchensDataset(root=root, split=split, **ds_cfg))
        elif name == "assembly101":
            datasets.append(Assembly101Dataset(root=root, split=split, **ds_cfg))
        elif name == "holoassist":
            datasets.append(HoloAssistDataset(root=root, split=split, **ds_cfg))
        else:
            raise ValueError(f"Unknown dataset: {name}")

    return datasets


def build_dataloader(
    datasets: list[BaseEgocentricDataset],
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    domain_probs: list[float] | None = None,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Build unified dataloader with domain-aware sampling."""
    concat_dataset = ConcatDatasetWithDomain(datasets)
    dataset_sizes = [len(d) for d in datasets]

    sampler = DomainAwareSampler(
        dataset_sizes=dataset_sizes,
        batch_size=batch_size,
        domain_probs=domain_probs,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    def collate_fn(batch):
        return {
            "video": torch.stack([b["video"] for b in batch]),
            "verb_label": torch.tensor([b["verb_label"] for b in batch]),
            "noun_label": torch.tensor([b["noun_label"] for b in batch]),
            "action_label": torch.tensor([b["action_label"] for b in batch]),
            "video_id": [b["video_id"] for b in batch],
            "domain_idx": torch.tensor([b["domain_idx"] for b in batch]),
            "domain_name": [b["domain_name"] for b in batch],
        }

    return DataLoader(
        concat_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )