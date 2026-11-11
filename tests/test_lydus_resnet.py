"""benchtoward/lydus-resnet1d: preprocessing (numpy) and the ported architecture (needs PyTorch; skipped without it)."""
import numpy as np
import pytest

from benchtoward.baselines import lydus_resnet as rn
from benchtoward.tasks import lydus


def test_resample_pad_gives_4096_samples_at_400_hz():
    t = np.arange(5000) / 500
    x = np.tile((np.sin(2 * np.pi * 1.0 * t) * 1000).astype(np.int16), (2, 8, 1))
    y = rn.resample_pad(x)
    assert y.shape == (2, 8, 4096) and y.dtype == np.float16
    assert np.all(y[..., :48] == 0) and np.all(y[..., -48:] == 0)       # 4,000 samples centred
    assert abs(float(y[0, 0].max()) - 1000 * rn.COUNTS_TO_MV) < 0.1


def test_lead_matrix_derives_the_challenge_lead_sets():
    x8 = np.random.default_rng(0).normal(size=(8, 10)).astype(np.float32)
    for lead_set in lydus.LEAD_SETS:
        assert np.allclose(rn.lead_matrix(lead_set) @ x8, lydus.select_leads(lydus.to_12_leads(x8), lead_set),
                           atol=1e-5)


def test_architecture_matches_ribeiro_2020_shapes():
    torch = pytest.importorskip("torch")
    model = rn.build_model(12, 25)
    x = torch.zeros(2, 12, 4096)
    assert model(x).shape == (2, 25)
    shapes = []
    h = torch.relu(model.bn(model.conv(x)))
    y = h
    for unit in model.units:
        h, y = unit(h, y)
        shapes.append(tuple(h.shape[1:]))
    assert shapes == [(128, 1024), (196, 256), (256, 64), (320, 16)]


def test_training_runs_on_cpu_and_writes_scores(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8, 4096)).astype(np.float16)
    y = (rng.random((30, 25)) < 0.3).astype(np.float32)
    split = np.array([0] * 20 + [1] * 5 + [2] * 5)
    settings = rn.TRAINING | {"max_epochs": 2, "batch_size": 8}
    result = rn.train(X, y, np.ones_like(y, bool), split, "2", tmp_path, device="cpu", settings=settings, log=lambda _: None)
    scores = np.load(tmp_path / "scores-2.npz")
    assert scores["test_scores"].shape == (5, 25) and result["epochs"] == 2
    assert ((scores["test_scores"] >= 0) & (scores["test_scores"] <= 1)).all()


def test_masked_bce_has_no_unknown_gradient():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 9.0]], requires_grad=True)
    y = torch.tensor([[1.0, 0.0]])
    loss = rn.masked_bce(logits, y, torch.tensor([[True, False]]))
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert logits.grad.tolist() == [[-0.5, 0.0]]


def test_pod_scores_require_current_labels_order_and_masks(tmp_path):
    import pandas as pd
    inst = pd.DataFrame({"row": [3, 1, 2], "split": ["train", "validation", "test"],
                         "targets": [[], [], []], "unknown_targets": [[], [], []]})
    from benchtoward.tasks.lydus_ecg12_dx import label_matrices
    y, known = label_matrices(inst)
    digest = rn.labels_digest(y, known, [0, 1, 2], inst["row"])
    args = {"val_rows": [1], "test_rows": [2], "val_scores": np.zeros((1, 25)), "test_scores": np.zeros((1, 25))}
    np.savez(tmp_path / "scores-12.npz", **args, labels_sha256=digest)
    assert rn.load_pod_scores(tmp_path, "12", inst)["test"][1].shape == (1, 25)
    inst.at[0, "unknown_targets"] = ["rhythm:A.fib"]
    with pytest.raises(ValueError, match="labels/order/masks"):
        rn.load_pod_scores(tmp_path, "12", inst)
