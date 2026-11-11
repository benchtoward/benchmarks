"""Reporting around the frozen provided-label scorer; no change to its primary metrics."""
import numpy as np

from benchtoward.tasks import lydus_labels as ll


def metrics(y, known, scores, groups, thresholds=None, n_boot=ll.N_BOOT, seed=ll.SEED):
    result = ll.metrics(y, known, scores, groups, thresholds=thresholds, n_boot=n_boot, seed=seed)
    diagnostic = tuple(t for t in ll.TARGETS if t not in ll.PRIMARY)
    report = ll.metrics(y, known, scores, groups, thresholds=thresholds, n_boot=n_boot, seed=seed,
                        targets=diagnostic)
    result["per_target"].update(report["per_target"])
    groups = np.asarray(groups)
    for k, target in enumerate(ll.TARGETS):
        pos = y[:, k] & known[:, k]
        result["per_target"][target] |= {
            "primary": target in ll.PRIMARY,
            "negatives": int((~y[:, k] & known[:, k]).sum()),
            "known_groups": len(set(groups[known[:, k]])),
            "positive_groups": len(set(groups[pos])),
        }
    result["bootstrap"]["macro_dropped_fraction"] = result["bootstrap"]["macro_dropped"] / n_boot if n_boot else None
    return result
