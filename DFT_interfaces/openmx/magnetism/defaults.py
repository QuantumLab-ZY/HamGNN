from __future__ import annotations

from typing import Any

from pymatgen.core.periodic_table import Element

from DFT_interfaces.openmx.magnetism.config import ConfigError
from DFT_interfaces.openmx.magnetism._defaults_data import PBE_dict, PAO_dict, basis_def_14, basis_def_19, basis_def_26, spin_set

DEFAULT_SPIN = spin_set
DEFAULT_PAO = PAO_dict
DEFAULT_PBE = PBE_dict
DEFAULT_SPIN_CONSTRAINT = {symbol: 0 for symbol in DEFAULT_PAO}
BASIS_DEFINITIONS = {14: basis_def_14, 19: basis_def_19, 26: basis_def_26}
REQUIRED_OVERRIDE_FIELDS = ("pao", "pbe", "spin", "spin_constraint", "basis")


def get_basis_definitions(nao_max: int):
    try:
        return BASIS_DEFINITIONS[int(nao_max)]
    except KeyError as exc:
        supported = ", ".join(str(key) for key in sorted(BASIS_DEFINITIONS))
        raise ConfigError(
            f"Unsupported nao_max={nao_max}. Supported values: {supported}."
        ) from exc


def resolve_species_settings(symbols, config: dict[str, Any] | None, nao_max: int = 26):
    overrides = config.get("overrides", {}) if config else {}
    resolved = {}
    for symbol in symbols:
        if symbol in resolved:
            continue

        override = overrides.get(symbol, {})
        if symbol not in DEFAULT_PAO and not override:
            raise ConfigError(
                f"No built-in OpenMX defaults for element '{symbol}'. Add species.overrides.{symbol} "
                "with pao, pbe, spin, spin_constraint, and basis."
            )

        if symbol not in DEFAULT_PAO:
            missing_fields = [field for field in REQUIRED_OVERRIDE_FIELDS if field not in override]
            if missing_fields:
                missing = ", ".join(missing_fields)
                raise ConfigError(
                    f"Override-only species '{symbol}' is missing required fields: {missing}."
                )

        atomic_number = Element(symbol).Z if symbol in DEFAULT_PAO else None
        basis = override.get("basis")
        if basis is None and atomic_number is not None:
            basis_array = get_basis_definitions(nao_max).get(atomic_number)
            basis = basis_array.tolist() if basis_array is not None else []

        spin = override.get("spin", DEFAULT_SPIN.get(symbol))
        if isinstance(spin, list):
            spin = list(spin)

        resolved[symbol] = {
            "pao": override.get("pao", DEFAULT_PAO.get(symbol)),
            "pbe": override.get("pbe", DEFAULT_PBE.get(symbol)),
            "spin": spin,
            "spin_constraint": override.get(
                "spin_constraint", DEFAULT_SPIN_CONSTRAINT.get(symbol, 0)
            ),
            "basis": basis,
        }

    return resolved
