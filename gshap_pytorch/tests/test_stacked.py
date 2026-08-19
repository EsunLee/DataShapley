import numpy as np
import torch

from gshap.config import ExperimentConfig, RuntimeConfig, TrainConfig
from gshap.data import DatasetBundle, create_split
from gshap.metrics import binary_auc, binary_auc_batch
from gshap.trainer import GShapTrainer


def _make_bundle() -> DatasetBundle:
    rng = np.random.default_rng(11)
    n = 2_048
    features = rng.normal(size=(n, 512)).astype(np.float32)
    # 1-D labels: the trainer contract (load_or_create_bundle reshapes to (-1))
    labels = (rng.random(n) < 0.5).astype(np.float32)
    train, test, lr_val = create_split(labels, test_size=48, lr_val_size=16, seed=0)
    return DatasetBundle(features, labels, train, test, lr_val)


def _make_trainer(bundle: DatasetBundle, tmp_path) -> GShapTrainer:
    config = ExperimentConfig(
        data_dir=tmp_path, results_dir=tmp_path / "results",
        train=TrainConfig(batch_size=128, iterations=2, learning_rate=0.1),
        runtime=RuntimeConfig(device="cpu", deterministic=True),
    )
    return GShapTrainer(bundle, config, 0.1)


def test_binary_auc_batch_matches_per_stream_exactly():
    torch.manual_seed(3)
    scores = torch.randn(4, 317)
    labels = torch.rand(317) < 0.42
    batched = binary_auc_batch(scores, labels)
    for stream in range(4):
        single = binary_auc(scores[stream], labels)
        assert batched[stream].item() == single.item()


def test_stacked_matches_sequential(tmp_path):
    trainer = _make_trainer(_make_bundle(), tmp_path)
    seeds = [101, 202]
    sequential = np.stack([trainer._run_permutation(seed)[0] for seed in seeds])
    stacked, _, _, _, _, _, _ = trainer._run_permutations_stacked(seeds)
    np.testing.assert_allclose(stacked, sequential, atol=1e-12, rtol=0)


def test_stacked_is_deterministic(tmp_path):
    trainer = _make_trainer(_make_bundle(), tmp_path)
    first = trainer._run_permutations_stacked([101, 202])[0]
    second = trainer._run_permutations_stacked([101, 202])[0]
    np.testing.assert_array_equal(first, second)
