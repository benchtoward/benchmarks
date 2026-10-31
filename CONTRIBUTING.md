# Contributing

## Report an evaluation problem

The most useful contribution is a report that something in a benchmark is wrong or unfair: a label rule, a split,
leakage between inputs and labels, a metric, a parser that marks right answers wrong, or a deviation from the
protocol a benchmark says it adopts. Open an issue with the **Evaluation problem** template and include the task and
version (for example `inspire-mortality@0`), what you expected, and how to see it (a command, a row, an instance id).

Released task versions are never changed silently. A confirmed problem is fixed in a new version, and the old
version's page says why it was replaced.

## Code

- Pull requests are welcome; the maintainers publish this repository from the code that runs BenchToward, so an
  accepted change is applied there and appears here with the next export.
- Tests come with every change (`uv run pytest -q`); tests do not use the network.
- Do not copy code from projects without a license that allows it. Several protocols BenchToward adopts are
  published without a license; benchmarks reimplement them from their description and pin the original commit.
  Code under a compatible license (MIT, BSD, Apache-2.0) may be vendored under `third_party/<name>/` with its
  license file and a note of what changed.
- Datasets are never committed. Builders download them from their sources.
