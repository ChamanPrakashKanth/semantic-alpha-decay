from train import run


def test_tiny_training_smoke():
    result = run(seed=0, mode="learned", steps=2, batch_size=8, keep_penalty=.01)
    assert "cross_entropy" in result["iid"]
    assert set(result["interventions"]) == {"zero_relevant", "shuffle", "random", "force_one"}
