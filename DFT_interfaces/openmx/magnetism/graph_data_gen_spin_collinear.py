import sys

from DFT_interfaces.openmx.magnetism.cli import main


def _has_user_mode_arg(argv: list[str]) -> bool:
    return any(arg == "--mode" or arg.startswith("--mode=") for arg in argv)


if __name__ == "__main__":
    if _has_user_mode_arg(sys.argv[1:]):
        raise SystemExit(
            "error: graph_data_gen_spin_collinear.py fixes --mode=collinear; use "
            "python -m DFT_interfaces.openmx.magnetism.cli pack-graph-data for manual mode selection."
        )
    raise SystemExit(main(["pack-graph-data", *sys.argv[1:], "--mode", "collinear"]))
