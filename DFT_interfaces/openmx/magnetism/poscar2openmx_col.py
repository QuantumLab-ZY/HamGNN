import sys

from DFT_interfaces.openmx.magnetism.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["convert-collinear", *sys.argv[1:]]))
