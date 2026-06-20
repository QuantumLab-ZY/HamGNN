from __future__ import annotations

import json
import subprocess
from pathlib import Path

from DFT_interfaces.openmx.magnetism.config import ConfigError


def discover_files(patterns):
    """Return sorted Path matches for one or more glob patterns."""
    if isinstance(patterns, (str, Path)):
        pattern_list = [patterns]
    else:
        pattern_list = list(patterns)

    matches: list[Path] = []
    for pattern in pattern_list:
        pattern_str = str(pattern)
        p = Path(pattern_str)
        if p.is_absolute():
            # For absolute paths, split into anchor and glob pattern parts
            parts = p.parts
            glob_idx = None
            for i, part in enumerate(parts):
                if any(c in part for c in "*?["):
                    glob_idx = i
                    break
            if glob_idx is not None:
                root = Path(*parts[:glob_idx])
                pattern_part = "/".join(parts[glob_idx:])
                matches.extend(root.glob(pattern_part))
            else:
                if p.is_dir():
                    matches.extend(p.iterdir())
                elif p.exists():
                    matches.append(p)
        else:
            matches.extend(Path().glob(pattern_str))

    sorted_matches = sorted({match.resolve() for match in matches})
    if not sorted_matches:
        raise ConfigError(f"No files matched pattern(s): {', '.join(str(pattern) for pattern in pattern_list)}")

    return sorted_matches


def ensure_output_dir(path):
    """Create and return an output directory Path."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def output_path_for(input_path, output_dir, suffix, prefix="", overwrite=False):
    """Return a derived output Path and reject collisions unless overwrite is true."""
    candidate = Path(output_dir) / f"{prefix}{Path(input_path).stem}{suffix}"
    if candidate.exists() and not overwrite:
        raise ConfigError(f"Refusing to overwrite existing output file '{candidate}'.")
    return candidate


def resolve_read_openmx(path):
    """Return an executable read_openmx Path from a file path or containing directory."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "read_openmx"

    if not candidate.is_file():
        raise ConfigError(f"read_openmx executable not found at '{candidate}'.")
    if not candidate.exists() or not candidate.stat().st_mode & 0o111:
        raise ConfigError(f"read_openmx path '{candidate}' is not executable.")

    return candidate


def load_hs_json(path):
    """Load and return an HS.json mapping with clear JSON/file errors."""
    hs_path = Path(path)
    try:
        with hs_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in HS.json '{hs_path}': {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read HS.json '{hs_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigError(f"HS.json '{hs_path}' must contain a JSON object.")

    return payload


def run_read_openmx(read_openmx, scfout_file, work_dir):
    """Run read_openmx in work_dir and return the parsed HS.json payload."""
    command = [str(read_openmx), str(scfout_file)]
    try:
        subprocess.run(
            command,
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        details = [
            f"Failed to run read_openmx command {command!r} for '{scfout_file}' in work_dir '{work_dir}'.",
        ]
        stdout = getattr(exc, "stdout", None)
        stderr = getattr(exc, "stderr", None)
        if stdout:
            details.append(f"stdout: {stdout}")
        if stderr:
            details.append(f"stderr: {stderr}")
        raise ConfigError(" ".join(details)) from exc

    return load_hs_json(Path(work_dir) / "HS.json")
