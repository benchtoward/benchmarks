"""Environment variable names. The project is BenchToward (owner, 2026-10-31): its variables are BENCHTOWARD_*, and
the MEDBENCH_* names they replace are still read as a fallback, so an existing .env (the site server's, the Mac's)
keeps working unchanged. Mapping: BENCHTOWARD_<X> replaces MEDBENCH_<X> for every <X> (README, "Environment").

Rules: the new name wins; an empty value counts as unset, as in the scripts' ${BENCHTOWARD_X:-${MEDBENCH_X:-...}}.
Code that sets a variable for a child sets the new name and drops the old one, so a stale old value never shadows
it. The fallback is removed at the work-host cutover (docs/design/2026-10-30-cloud-automation-design.md)."""
import os
from collections.abc import Mapping, MutableMapping

NEW = "BENCHTOWARD_"
OLD = "MEDBENCH_"


def names(suffix: str) -> tuple[str, str]:
    """(new name, old name) of one variable, e.g. names("DB_URL")."""
    return NEW + suffix, OLD + suffix


def getenv(suffix: str, default: str | None = None, env: Mapping[str, str] | None = None) -> str | None:
    """The first non-empty of BENCHTOWARD_<suffix> and MEDBENCH_<suffix>, else default."""
    env = os.environ if env is None else env
    for name in names(suffix):
        if env.get(name):
            return env[name]
    return default


def setenv(env: MutableMapping[str, str], suffix: str, value: str) -> None:
    env[NEW + suffix] = value
    env.pop(OLD + suffix, None)


def unsetenv(env: MutableMapping[str, str], suffix: str) -> None:
    for name in names(suffix):
        env.pop(name, None)


def both(*suffixes: str) -> tuple[str, ...]:
    """Every name the variables may have, for lists of names to strip or redact."""
    return tuple(name for suffix in suffixes for name in names(suffix))


def is_ours(name: str) -> bool:
    return name.startswith((NEW, OLD))
