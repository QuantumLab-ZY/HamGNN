from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when a user-provided magnetism configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cannot parse config file '{config_path}': {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read config file '{config_path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Config file '{config_path}' must contain a YAML mapping.")

    return config


def require_sections(config: dict[str, Any], sections: Iterable[str]) -> None:
    missing = [section for section in sections if section not in config]
    if missing:
        raise ConfigError(f"Missing required section(s): {', '.join(missing)}")


def merge_cli_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    merged = deepcopy(config)

    for section_name in ("runtime", "inputs", "outputs", "graph_data"):
        section = merged.get(section_name)
        if section is not None and not isinstance(section, dict):
            raise ConfigError(f"Config section '{section_name}' must be a mapping.")

    runtime = merged.setdefault("runtime", {})
    inputs = merged.setdefault("inputs", {})
    outputs = merged.setdefault("outputs", {})
    graph_data = merged.setdefault("graph_data", {})

    if overrides.get("input_pattern") is not None:
        inputs["patterns"] = [overrides["input_pattern"]]
    if overrides.get("output_dir") is not None:
        outputs["directory"] = overrides["output_dir"]
    if overrides.get("read_openmx") is not None:
        graph_data["read_openmx"] = overrides["read_openmx"]
    if overrides.get("nao_max") is not None:
        graph_data["nao_max"] = overrides["nao_max"]
    if overrides.get("workers") is not None:
        runtime["workers"] = overrides["workers"]
    if overrides.get("dry_run") is not None:
        runtime["dry_run"] = overrides["dry_run"]
    if overrides.get("skip_errors") is not None:
        runtime["skip_errors"] = overrides["skip_errors"]

    return merged
