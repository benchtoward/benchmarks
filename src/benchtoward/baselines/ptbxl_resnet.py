"""benchtoward/ptbxl-resnet1d: the LYDUS 1-D ResNet (benchtoward.baselines.lydus_resnet) trained on PTB-XL 1.0.3 for
ptbxl-superdiagnostic@0.

Same architecture and training recipe as benchtoward/lydus-resnet1d (Ribeiro et al., Nat Commun 2020;11:1760,
antonior92/automatic-ecg-diagnosis @ 4039348, MIT): 12 leads built from the stored independent leads I, II, V1-V6,
resampled from 500 to 400 Hz and zero-padded to 4,096 samples, Adam 1e-3, batch 64, plateau schedule, early stopping
on validation loss, at most 70 epochs, seed 0. Differences: the input scale (PTB-XL stores 1 uV per unit, so x 0.001
gives mV; LYDUS counts are about 4.88 uV), five outputs (the task's superclasses, all known: the masked loss masks
nothing), selection on fold 9 only, and no thresholds (the task's metric is macro AUROC on scores).

Pod entry point: `python -m benchtoward.baselines.ptbxl_resnet --release <PTB-XL root> --labels labels.npz --out ...
--work ...`; it needs numpy, scipy, pandas and torch, and only this file, lydus_resnet.py, lydus.py, lydus_labels.py,
lydus_reporting.py and ptbxl.py of BenchToward.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from benchtoward.baselines import lydus_resnet
from benchtoward.tasks import ptbxl

CLASSES = ptbxl.SUPERCLASSES
UV_TO_MV = 0.001
TRAINING = dict(lydus_resnet.TRAINING)
LEAD_SET = "12"
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


def resample_pad_uv(x8: np.ndarray) -> np.ndarray:
    """(n, 8, 5000) int16 uV at 500 Hz -> (n, 8, 4096) float16 mV at 400 Hz, zero-padded on both sides."""
    from scipy.signal import resample_poly

    y = resample_poly(np.asarray(x8, dtype=np.float32), lydus_resnet.OUT_HZ // 100, lydus_resnet.IN_HZ // 100,
                      axis=-1) * UV_TO_MV
    pad = lydus_resnet.LENGTH - y.shape[-1]
    out = np.zeros((*y.shape[:-1], lydus_resnet.LENGTH), dtype=np.float16)
    out[..., pad // 2:pad // 2 + y.shape[-1]] = y
    return out


def labels_digest(ecg_ids, split, y) -> str:
    payload = {"classes": list(CLASSES), "ecg_id": np.asarray(ecg_ids).astype(int).tolist(),
               "split": np.asarray(split).astype(int).tolist(), "y": np.asarray(y, dtype=bool).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def label_arrays(inst) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ecg_id, split code (0 train, 1 validation, 2 test) and (n, 5) labels in the instances' order."""
    index = {c: k for k, c in enumerate(CLASSES)}
    y = np.zeros((len(inst), len(CLASSES)), dtype=bool)
    for r, targets in enumerate(inst["targets"]):
        y[r, [index[t] for t in targets]] = True
    split = inst["split"].map(SPLIT_CODES).to_numpy(np.int8)
    return inst["ecg_id"].to_numpy(np.int64), split, y


def labels_file(inst, filenames: dict) -> bytes:
    """What the pod trains on, in the instances' order: ecg_id, the 500 Hz record path, split and labels. No patient
    ids, no report text."""
    ecg_ids, split, y = label_arrays(inst)
    buf = io.BytesIO()
    np.savez(buf, ecg_id=ecg_ids, filename=np.array([filenames[int(i)] for i in ecg_ids]), split=split, y=y,
             classes=np.array(CLASSES), labels_sha256=labels_digest(ecg_ids, split, y))
    return buf.getvalue()


def build_cache(release_root: Path, filenames, path: Path, chunk: int = 1000) -> Path:
    """Read the records in `filenames` order, keep I, II, V1-V6 and write (n, 8, 4096) float16 mV to `path`."""
    filenames = list(filenames)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                    shape=(len(filenames), 8, lydus_resnet.LENGTH))
    for s in range(0, len(filenames), chunk):
        raw = np.stack([ptbxl.stored_eight(ptbxl.read_record(release_root, f)) for f in filenames[s:s + chunk]])
        out[s:s + len(raw)] = resample_pad_uv(raw)
    out.flush()
    del out
    return Path(path)


def train_cache(X400, y, split, out_dir: Path, device: str = "cuda", settings: dict = TRAINING, log=print,
                labels_sha256: str = "") -> dict:
    """lydus_resnet.train with every label known and the 12-lead set; writes scores-12.npz to out_dir."""
    y = np.asarray(y, dtype=np.float32)
    return lydus_resnet.train(X400, y, np.ones_like(y, dtype=bool), np.asarray(split), LEAD_SET, Path(out_dir),
                              device=device, settings=settings, log=log, labels_sha256=labels_sha256)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", required=True, help="the extracted PTB-XL 1.0.3 folder")
    parser.add_argument("--labels", required=True, help="labels.npz from ptbxl_resnet.labels_file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True, help="folder for the resampled signal cache")
    args = parser.parse_args(argv)
    out, work = Path(args.out), Path(args.work)
    labels = np.load(args.labels)
    if str(labels["labels_sha256"]) != labels_digest(labels["ecg_id"], labels["split"], labels["y"]):
        raise SystemExit("labels.npz does not match its digest")
    progress = work / "progress.json"
    cache = work / "x400.npy"
    if not cache.exists():
        progress.write_text(json.dumps({"state": "reading records"}))
        build_cache(Path(args.release), [str(f) for f in labels["filename"]], cache)
    progress.write_text(json.dumps({"state": "training"}))
    train_cache(np.load(cache), labels["y"], labels["split"], out, labels_sha256=str(labels["labels_sha256"]),
                log=lambda line: (work / "train.log").open("a").write(line + "\n"))
    progress.write_text(json.dumps({"state": "done"}))


def load_pod_scores(folder: Path, inst) -> tuple[np.ndarray, np.ndarray]:
    """Test rows (positions in the instances) and their (n, 5) scores; refuses scores trained on other labels."""
    data = np.load(Path(folder) / f"scores-{LEAD_SET}.npz")
    ecg_ids, split, y = label_arrays(inst)
    if str(data["labels_sha256"]) != labels_digest(ecg_ids, split, y):
        raise ValueError("ResNet scores were trained on other labels, order or splits; train again")
    rows = np.asarray(data["test_rows"])
    if not np.all(split[rows] == SPLIT_CODES["test"]) or len(rows) != int((split == SPLIT_CODES["test"]).sum()):
        raise ValueError("scores-12.npz: test rows are not the test split")
    return rows, np.asarray(data["test_scores"], dtype=float)


def _macro(y, scores, rows):
    from benchtoward.tasks.lydus_labels import auroc

    values = [auroc(y[rows, k], scores[rows, k]) for k in range(y.shape[1])]
    return float(np.mean(values)) if all(np.isfinite(values)) else None


def paired_difference(y, strong, weak, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Macro AUROC of `strong` minus `weak` on the same records, with a 95% interval from whole-patient bootstrap
    draws shared by both (draws where a class has one label only are left out and counted)."""
    y, strong, weak = np.asarray(y, dtype=bool), np.asarray(strong, float), np.asarray(weak, float)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    everything = np.arange(len(y))
    value = _macro(y, strong, everything) - _macro(y, weak, everything)
    draws, dropped = [], 0
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        a, b = _macro(y, strong, rows), _macro(y, weak, rows)
        if a is None or b is None:
            dropped += 1
            continue
        draws.append(a - b)
    draws = np.asarray(draws)
    return {"value": float(value), "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)), "share_at_or_below_zero": float(np.mean(draws <= 0)),
            "draws": int(len(draws)), "dropped": dropped, "seed": seed, "by": "patient_id"}


def sop_verdict(best: dict, prior: float, weak: dict, strong: dict, format_failures: float,
                ceiling: float = 1.0) -> dict:
    """The benchmark check of docs/sop/benchmarks-and-models.md for a macro AUROC task (value, ci_low, ci_high)."""
    H = (ceiling - best["value"]) / (ceiling - prior)
    headroom = "fail" if H < 0.05 or best["ci_high"] >= ceiling else ("warning" if H < 0.10 else "pass")
    beats = all(m["ci_low"] > prior and m["value"] >= 0.55 for m in (weak, strong))
    separated = strong["ci_low"] > weak["ci_high"]
    result = {
        "headroom": {"H": round(H, 4), "best_ci_high": best["ci_high"], "result": headroom},
        "beats_prior": {"weak_ci_low": weak["ci_low"], "strong_ci_low": strong["ci_low"],
                        "result": "pass" if beats else "fail"},
        "separation": {"strong_ci_low": strong["ci_low"], "weak_ci_high": weak["ci_high"],
                       "result": "pass" if separated else "fail"},
        "format": {"failures": format_failures, "result": "pass" if format_failures < 0.10 else "fail"},
    }
    failed = [k for k, v in result.items() if v["result"] == "fail"]
    result["verdict"] = "fail" if failed else "pass"
    result["failed"] = failed
    return result


if __name__ == "__main__":
    main()
