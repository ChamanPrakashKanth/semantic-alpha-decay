from train import evaluate, run
from data import BatchFactory, VOCAB_SIZE
from sadt.model import TinySADT


def test_tiny_training_smoke():
    result = run(seed=0, mode="learned", steps=2, batch_size=8, keep_penalty=.01)
    assert "cross_entropy" in result["iid"]
    assert set(result["interventions"]) == {"zero_relevant", "shuffle", "random", "force_one"}


def test_diagnostic_metrics_are_logged():
    metrics = evaluate(TinySADT(VOCAB_SIZE, mode="constrained"), BatchFactory(0),
                       "unseen_layout", batches=1, batch_size=8)
    assert metrics["delta_D"] == metrics["D_rel"] - metrics["D_irr"]
