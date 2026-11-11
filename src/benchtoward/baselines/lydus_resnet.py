"""benchtoward/lydus-resnet1d: a 1-D ResNet trained from scratch on the LYDUS train split (lydus-ecg12-dx@0).

The architecture is that of Ribeiro et al., Automatic diagnosis of the 12-lead ECG using a deep neural network, Nat
Commun 2020;11:1760 (antonior92/automatic-ecg-diagnosis @ 403934867cea25a85b54faca025ac65b1ac18a45, model.py, MIT,
Copyright (c) 2020 Antonio Horta Ribeiro; licence in third_party/automatic-ecg-diagnosis/), ported from Keras to
PyTorch: a convolution with 64 filters, then four pre-activation residual units (128, 196, 256, 320 filters; 1,024,
256, 64, 16 samples out; kernel 16; dropout 0.2; max-pool and 1x1 convolution on the skip path), flattened into a
sigmoid layer, 25 outputs for the dataset-provided targets. Training follows its train.py: masked binary cross-entropy on known entries, Adam at
1e-3, batch 64, learning rate x0.1 after 7 epochs without a lower validation loss (down to 1e-5), stop after 9, at
most 70 epochs; the weights of the lowest validation loss are kept. Its released weights (six classes, trained on
CODE) are not used. As in its preprocessing, records are resampled to 400 Hz and zero-padded to 4,096 samples (both
sides). The input of a lead set is its leads only; age and sex are not used. Thresholds: per class, the highest
validation F-measure, 0.5 for a class with fewer than 10 validation positives (as the gradient-boosting
baseline).

Runs on a GPU with PyTorch; `python -m benchtoward.baselines.lydus_resnet --help`.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchtoward.tasks.lydus_reporting import metrics as classification_metrics

COMMIT = "403934867cea25a85b54faca025ac65b1ac18a45"
IN_HZ, OUT_HZ, LENGTH = 500, 400, 4096
COUNTS_TO_MV = 0.00488   # amplitude scale for the input only (about 4.88 uV per count); batch norm follows
TRAINING = {"lr": 1e-3, "batch_size": 64, "max_epochs": 70, "plateau_patience": 7, "plateau_factor": 0.1,
            "min_lr": 1e-5, "early_stop_patience": 9, "min_delta": 1e-5, "dropout": 0.2, "kernel_size": 16,
            "seed": 0}


def resample_pad(x8: np.ndarray) -> np.ndarray:
    """(n, 8, 5000) counts at 500 Hz -> (n, 8, 4096) float16 mV at 400 Hz, zero-padded on both sides."""
    from scipy.signal import resample_poly

    y = resample_poly(np.asarray(x8, dtype=np.float32), OUT_HZ // 100, IN_HZ // 100, axis=-1) * COUNTS_TO_MV
    pad = LENGTH - y.shape[-1]
    out = np.zeros((*y.shape[:-1], LENGTH), dtype=np.float16)
    out[..., pad // 2:pad // 2 + y.shape[-1]] = y
    return out


def lead_matrix(lead_set: str) -> np.ndarray:
    """(leads of the set, 8): rows that turn the stored leads I, II, V1-V6 into the set's leads."""
    from benchtoward.tasks import lydus

    eye = np.eye(8, dtype=np.float32)
    i, ii = eye[0], eye[1]
    rows = {"I": i, "II": ii, "III": ii - i, "aVR": -(i + ii) / 2, "aVL": i - ii / 2, "aVF": ii - i / 2,
            **{f"V{k}": eye[1 + k] for k in range(1, 7)}}
    return np.stack([rows[name] for name in lydus.LEAD_SETS[lead_set]])


def build_model(n_leads: int, n_classes: int, kernel_size: int = 16, dropout: float = 0.2):
    import torch
    from torch import nn

    class ResidualUnit(nn.Module):
        def __init__(self, n_in, n_out, downsample):
            super().__init__()
            self.downsample = downsample
            self.skip_pool = nn.MaxPool1d(downsample, stride=downsample) if downsample > 1 else nn.Identity()
            self.skip_conv = nn.Conv1d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
            self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, padding="same", bias=False)
            self.bn1 = nn.BatchNorm1d(n_out)
            self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, stride=downsample, bias=False)
            self.bn2 = nn.BatchNorm1d(n_out)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, y):
            y = self.skip_conv(self.skip_pool(y))
            x = self.drop(torch.relu(self.bn1(self.conv1(x))))
            # Keras 'same' padding with a stride: pad so that the output has ceil(n / stride) samples
            n = x.shape[-1]
            out = -(-n // self.downsample)
            total = max((out - 1) * self.downsample + kernel_size - n, 0)
            x = self.conv2(nn.functional.pad(x, (total // 2, total - total // 2)))
            x = x + y
            y = x
            x = self.drop(torch.relu(self.bn2(x)))
            return x, y

    class ResNet1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(n_leads, 64, kernel_size, padding="same", bias=False)
            self.bn = nn.BatchNorm1d(64)
            self.units = nn.ModuleList([ResidualUnit(64, 128, 4), ResidualUnit(128, 196, 4),
                                        ResidualUnit(196, 256, 4), ResidualUnit(256, 320, 4)])
            self.dense = nn.Linear(320 * 16, n_classes)

        def forward(self, x):
            x = torch.relu(self.bn(self.conv(x)))
            y = x
            for unit in self.units:
                x, y = unit(x, y)
            return self.dense(x.flatten(1))        # logits; sigmoid in the loss and at prediction

    model = ResNet1d()
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)       # Keras he_normal
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    return model


def masked_bce(logits, labels, known):
    """Mean BCE over known target entries; unknown entries contribute no gradient."""
    import torch
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * known).sum() / known.sum().clamp(min=1)


def labels_digest(y, known, split, rows) -> str:
    import hashlib
    from benchtoward.tasks.lydus_labels import TARGETS
    payload = {"targets": TARGETS, "row": np.asarray(rows).tolist(), "split": np.asarray(split).tolist(),
               "y": np.asarray(y, dtype=bool).tolist(), "known": np.asarray(known, dtype=bool).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def train(X400: np.ndarray, y: np.ndarray, known: np.ndarray, split: np.ndarray, lead_set: str, out_dir: Path, device: str = "cuda",
          settings: dict = TRAINING, log=print, labels_sha256: str | None = None) -> dict:
    """Train on split == 0, select on split == 1; write scores of validation and test (split == 2) records."""
    import torch

    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    A = torch.tensor(lead_matrix(lead_set), device=device)
    model = build_model(A.shape[0], y.shape[1], settings["kernel_size"], settings["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    # Keras ReduceLROnPlateau(patience=7, min_delta=1e-4) lowers the rate once 7 epochs pass without an absolute
    # improvement of 1e-4; PyTorch lowers it after `patience` + 1 such epochs, hence patience - 1.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=settings["plateau_factor"],
                                                       patience=settings["plateau_patience"] - 1,
                                                       threshold=1e-4, threshold_mode="abs",
                                                       min_lr=settings["min_lr"])
    loss_fn = masked_bce
    idx = {k: np.flatnonzero(split == k) for k in (0, 1, 2)}
    Y = torch.tensor(y, dtype=torch.float32)
    K = torch.tensor(known, dtype=torch.bool)

    def batches(rows, shuffle):
        rows = np.random.permutation(rows) if shuffle else rows
        for s in range(0, len(rows), settings["batch_size"]):
            b = rows[s:s + settings["batch_size"]]
            x = torch.tensor(X400[b], device=device).float()
            yield torch.einsum("lc,bcn->bln", A, x), Y[b].to(device), K[b].to(device)

    def predict(rows):
        model.eval()
        out, total, count = [], 0.0, 0
        with torch.no_grad():
            for x, t, m in batches(rows, False):
                logits = model(x)
                total += loss_fn(logits, t, m).item() * int(m.sum())
                count += int(m.sum())
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out), total / max(count, 1)

    best, best_state, waited, history = np.inf, None, 0, []
    for epoch in range(settings["max_epochs"]):
        model.train()
        start, total, count = time.time(), 0.0, 0
        for x, t, m in batches(idx[0], True):
            opt.zero_grad()
            loss = loss_fn(model(x), t, m)
            loss.backward()
            opt.step()
            total += loss.item() * int(m.sum())
            count += int(m.sum())
        _, val_loss = predict(idx[1])
        sched.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": total / max(count, 1), "val_loss": val_loss,
                        "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - start, 1)})
        log(json.dumps(history[-1]))
        if val_loss < best - settings["min_delta"]:
            best, waited = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= settings["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    val_scores, _ = predict(idx[1])
    test_scores, _ = predict(idx[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"scores-{lead_set}.npz", val_rows=idx[1], val_scores=val_scores, test_rows=idx[2],
             test_scores=test_scores, labels_sha256=labels_sha256 or "")
    torch.save(best_state, out_dir / f"weights-{lead_set}.pt")
    result = {"lead_set": lead_set, "best_val_loss": best, "epochs": len(history), "history": history,
              "labels_sha256": labels_sha256, "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else device}
    (out_dir / f"history-{lead_set}.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signals", required=True, help="the release's 500 Hz NPZ")
    parser.add_argument("--labels", required=True, help="npz: row (release row), split (0/1/2), y and known (n, 25), targets, labels_sha256")
    parser.add_argument("--lead-sets", default="12,6,4,3,2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True, help="folder for the resampled signal cache")
    args = parser.parse_args(argv)
    out, work = Path(args.out), Path(args.work)
    progress = work / "progress.json"
    labels = np.load(args.labels)
    cache = work / "x400.npy"
    if not cache.exists():
        from benchtoward.tasks import lydus
        values = lydus.open_signals_file(Path(args.signals)).values
        rows = labels["row"]
        X400 = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16, shape=(len(rows), 8, LENGTH))
        for s in range(0, len(rows), 2000):
            X400[s:s + 2000] = resample_pad(np.asarray(values[rows[s:s + 2000]]))
        X400.flush()
        del X400
    X400 = np.load(cache)
    for lead_set in args.lead_sets.split(","):
        progress.write_text(json.dumps({"lead_set": lead_set, "state": "training"}))
        train(X400, labels["y"].astype(np.float32), labels["known"].astype(bool), labels["split"], lead_set, out,
              labels_sha256=str(labels["labels_sha256"]), log=lambda line: (work / "train.log").open("a").write(line + "\n"))
    progress.write_text(json.dumps({"state": "done"}))


if __name__ == "__main__":
    main()


def load_pod_scores(folder: Path, lead_set: str, inst):
    """Validation and test scores of one lead set from the pod outputs, aligned to the instances (release rows)."""
    data = np.load(Path(folder) / f"scores-{lead_set}.npz")
    from benchtoward.tasks.lydus_ecg12_dx import label_matrices
    y, known = label_matrices(inst)
    expected = labels_digest(y, known, inst["split"].map({"train": 0, "validation": 1, "test": 2}).to_numpy(),
                             inst["row"].to_numpy())
    if str(data.get("labels_sha256", "")) != expected:
        raise ValueError("ResNet scores were trained on other labels/order/masks; train again")
    split = inst["split"].to_numpy(object)
    out = {}
    for part, key in (("validation", "val"), ("test", "test")):
        # the pod indexes records by their position in the label file, which is the instances' order
        idx = data[f"{key}_rows"]
        if not np.all(split[idx] == part):
            raise ValueError(f"scores-{lead_set}.npz: {key} rows are not the {part} split")
        out[part] = (idx, data[f"{key}_scores"])
    return out


def evaluate_lead_set(folder: Path, lead_set: str, inst, n_boot: int) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """Thresholds from validation F-measure; metrics on test. Returns (metrics, outputs, scores, test rows)."""
    from benchtoward.tasks import lydus_labels as ll
    from benchtoward.baselines.lydus_ecg import f1_thresholds
    from benchtoward.tasks.lydus_ecg12_dx import label_matrices

    y, known = label_matrices(inst)
    parts = load_pod_scores(folder, lead_set, inst)
    v_idx, v_scores = parts["validation"]
    t_idx, t_scores = parts["test"]
    thresholds = f1_thresholds(y[v_idx], known[v_idx], v_scores)
    outputs = t_scores >= thresholds
    groups = inst["group"].to_numpy(object)[t_idx]
    metrics = classification_metrics(y[t_idx], known[t_idx], t_scores, groups,
                         thresholds=dict(zip(ll.TARGETS, map(float, thresholds))), n_boot=n_boot) | {"lead_set": lead_set}
    return metrics, outputs, t_scores, t_idx
