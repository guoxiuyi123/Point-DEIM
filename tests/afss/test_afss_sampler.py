import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data.afss_sampler import AFSSDistributedSampler
from engine.data.dataloader import DataLoader
from engine.data.dataset.afss_dataset import AFSSDataset
from engine.misc import dist_utils


class DummyDataset:
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return idx


def test_afss_sampler_tracks_active_subset_changes_between_epochs():
    dataset = AFSSDataset(dataset=DummyDataset())
    sampler = AFSSDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)

    assert list(iter(sampler)) == [0, 1, 2, 3, 4, 5, 6, 7]

    dataset.set_active_indices([7, 3, 1])
    sampler.set_epoch(2)

    assert list(iter(sampler)) == [0, 1, 2]


def test_afss_sampler_splits_current_active_subset_across_ranks():
    dataset = AFSSDataset(dataset=DummyDataset())
    dataset.set_active_indices([0, 1, 2, 3, 4, 5])

    rank0 = AFSSDistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)
    rank1 = AFSSDistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False)

    merged = list(iter(rank0)) + list(iter(rank1))
    assert sorted(merged) == [0, 1, 2, 3, 4, 5]


def test_warp_loader_uses_dataset_specific_sampler_hook(monkeypatch):
    dataset = AFSSDataset(dataset=DummyDataset())
    dataset.set_active_indices([7, 3, 1])
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
    loader.shuffle = False

    monkeypatch.setattr(dist_utils, "is_dist_available_and_initialized", lambda: True)
    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 0)

    wrapped = dist_utils.warp_loader(loader, shuffle=False)

    assert isinstance(wrapped.sampler, AFSSDistributedSampler)
    samples = [wrapped.dataset[idx] for idx in list(iter(wrapped.sampler))]
    assert samples == [7, 3, 1]
