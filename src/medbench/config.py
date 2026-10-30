import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    db_url: str = "sqlite:///data/medbench.db"
    data_dir: Path = Path("./data")
    # LLM serving routes: provider name -> (base URL, API key), from registry/providers.yaml and the environment.
    providers: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    audio_providers: tuple[str, ...] = ()  # routes allowed to receive dataset audio (`audio: true` in providers.yaml)
    # Request fields a route needs for a task protocol (`request:` in providers.yaml), e.g. how to switch thinking off.
    route_requests: dict[str, dict] = field(default_factory=dict)
    admin_password: str | None = None  # /admin is disabled when unset
    daily_goal: int = 100  # human visitors per day shown as the target on /admin
    analytics_secret: str | None = None  # salt for visitor hashes; generated into data_dir when unset
    agent_token: str | None = None  # shared secret for POST /v1/agent/events (paper agent -> site)
    site_url: str = "https://medbench.kr"
    llm_wiki_dir: Path | None = None  # paper archive the paper agent reads (LLM_WIKI_DIR)
    khdp_token: str | None = None  # KHDP Gateway personal access token; sent only to api.khdp.net
    allow_shared_files: bool = False  # allow third-party code on a compute account that also holds other data


def get_settings() -> Settings:
    env = os.environ
    defaults = Settings()
    return Settings(
        db_url=env.get("MEDBENCH_DB_URL") or defaults.db_url,
        data_dir=Path(env.get("MEDBENCH_DATA_DIR") or defaults.data_dir),
        providers=_providers(env),
        audio_providers=_audio_providers(),
        route_requests=_route_requests(),
        admin_password=env.get("MEDBENCH_ADMIN_PASSWORD") or None,
        daily_goal=int(env.get("MEDBENCH_DAILY_GOAL") or defaults.daily_goal),
        analytics_secret=env.get("MEDBENCH_ANALYTICS_SECRET") or None,
        agent_token=env.get("MEDBENCH_AGENT_TOKEN") or None,
        site_url=(env.get("MEDBENCH_SITE_URL") or defaults.site_url).rstrip("/"),
        llm_wiki_dir=Path(env["LLM_WIKI_DIR"]).expanduser() if env.get("LLM_WIKI_DIR") else None,
        khdp_token=env.get("KHDP_TOKEN") or None,
        allow_shared_files=env.get("MEDBENCH_ALLOW_SHARED_FILES", "") in ("1", "true", "yes"),
    )


def provider_routes() -> dict[str, dict]:
    """registry/providers.yaml as written: route name -> its entry ({} without the file). The file is part of the
    deployment, not of the evaluation code."""
    path = registry_dir() / "providers.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def _providers(env) -> dict[str, tuple[str, str | None]]:
    """Each entry of registry/providers.yaml: base_url, key_env (variable holding the API key), optional base_env
    (variable that overrides base_url)."""
    routes = {}
    for name, route in provider_routes().items():
        base = env.get(route.get("base_env") or "") or route["base_url"]
        routes[name] = (base.rstrip("/"), env.get(route["key_env"]) or None)
    return routes


def _audio_providers() -> tuple[str, ...]:
    return tuple(name for name, route in provider_routes().items() if route.get("audio"))


def _route_requests() -> dict[str, dict]:
    return {name: route["request"] for name, route in provider_routes().items() if route.get("request")}


def registry_dir() -> Path:
    return Path(os.environ.get("MEDBENCH_REGISTRY_DIR", "registry"))
