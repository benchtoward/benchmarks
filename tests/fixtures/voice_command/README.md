# Fixtures for kor-surgical-robot-voice-command@0

Real items, not made up: verbatim rows of the Korean Surgical Robot Voice-Command Benchmark 1.0.0 (Lee H., Seoul
National University Hospital, KHDP, doi:10.23421/2026.03507), licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). The rows are unchanged excerpts; only the selection below was made.

The overheard rows come from the Synthetic K-OR Speech Corpus v1.0 (KHDP,
doi:10.23421/2026.0773, CC BY 4.0). Rows whose `kor.source` is `mmor_translated` (here kor-MMO-M00547,
kor-MMO-M01185 and kor-MMO-M00961) are derived from MM-OR (Özsoy et al., CVPR 2025; Apache-2.0) and keep that
attribution.

These are scored test items with gold labels, so the set is kept to what the parity tests need:

- `clean.jsonl`: 19 items of `benchmark/commands_v0.1.jsonl` in 11 whole command groups (one or two per command
  type; alternative sequences, commands that depend on the robot's previous action, a group with 4 wording variants).
- `asr_generic.jsonl`: the same 19 items from `benchmark/commands_v0.1_asr_generic.jsonl`.
- `overheard.jsonl`: 6 items of `benchmark/kor_overheard_v0.1.jsonl` (rows 1-5 and 12 of the file).
- `scenes_v0.1.json`, `rules_v0.1.txt`: the release files, unchanged.
- `raw.json`: replies of three models to these items from the dataset author's 2026-10-26 runs (reply text and
  error only): `strong` Qwen3.8 Flash Next 180B-A6B, `mid` Gemma 4 26B-A4B, `weak` MedGemma 1.5 4B.
- `expected.json`: what the dataset author's original evaluation code returns on these inputs: system prompts of
  all 8 scenes, messages, the JSON schema, parsed replies, per-item verdicts, summary rows and paired comparisons.
  `tests/test_voice_command.py` requires the port to give the same values.
