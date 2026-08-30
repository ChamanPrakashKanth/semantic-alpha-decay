import torch

from gru_linear_small_model.config import ExperimentConfig
from gru_linear_small_model.data import RelationChainGenerator
from gru_linear_small_model.diagnostics import rescue_damage
from gru_linear_small_model.models import GRUBaseline
from gru_linear_small_model.training import train_models_shared_data


class RecordingGRU(GRUBaseline):
    def __init__(self):
        super().__init__(28, 24, 8, 8)
        self.seen = []

    def forward(self, tokens, mask=None, return_diagnostics=False):
        self.seen.append(tokens.detach().clone())
        return super().forward(tokens, mask, return_diagnostics)


def test_all_models_receive_identical_batches():
    a, b = RecordingGRU(), RecordingGRU()
    config = ExperimentConfig.smoke()
    config.steps = 3
    config.batch_size = 4
    train_models_shared_data(
        {"a": a, "b": b}, RelationChainGenerator(24, 9), config, torch.device("cpu")
    )
    assert len(a.seen) == len(b.seen) == 3
    assert all(torch.equal(x, y) for x, y in zip(a.seen, b.seen))


def test_rescue_damage_identity():
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    reference = torch.tensor([0, 9, 2, 9, 4, 9, 6, 9])
    candidate = torch.tensor([0, 1, 9, 3, 4, 9, 9, 7])
    result = rescue_damage(reference, candidate, targets)
    assert result["identity_error"] < 1e-7
