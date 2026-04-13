import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data.dataset.afss_dataset import AFSSDataset


class DummyDataset:
    def __init__(self, size=5):
        self.size = size
        self.category2name = {0: "cls"}
        self.label2category = {0: 0}
        self.epoch = -1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor([idx]), {"idx": torch.tensor([idx]), "image_id": torch.tensor([100 + idx])}

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_afss_dataset_defaults_to_all_indices_and_passthrough_attributes():
    base = DummyDataset(size=5)
    dataset = AFSSDataset(dataset=base)

    assert len(dataset) == 5
    sample, target = dataset[3]
    assert sample.item() == 3
    assert target["idx"].item() == 3
    assert dataset.category2name == {0: "cls"}
    assert dataset.label2category == {0: 0}


def test_afss_dataset_uses_active_indices_and_forwards_epoch():
    base = DummyDataset(size=5)
    dataset = AFSSDataset(dataset=base)
    dataset.set_active_indices([4, 1])
    dataset.set_epoch(7)

    first_sample, first_target = dataset[0]
    second_sample, second_target = dataset[1]

    assert len(dataset) == 2
    assert first_sample.item() == 4
    assert first_target["idx"].item() == 4
    assert second_sample.item() == 1
    assert second_target["idx"].item() == 1
    assert base.epoch == 7
