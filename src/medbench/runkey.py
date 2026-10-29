import hashlib
import json


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def run_key(model_digest: str, task_spec_hash: str, harness_version: str, config: dict) -> str:
    payload = canonical_json([model_digest, task_spec_hash, harness_version, config])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
