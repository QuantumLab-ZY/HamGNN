from pathlib import Path


def test_hamgnn_owned_sources_use_lightning_pytorch():
    roots = [Path("hamgnn"), Path("Uni-HamGNN"), Path("tests")]
    files = [
        path
        for root in roots
        for path in root.rglob("*.py")
        if path.name != "test_lightning2_imports.py"
    ]
    files = [
        path
        for path in files
        if "hamgnn/toolbox/nequip" not in path.as_posix()
    ]
    stale = [str(path) for path in files if "pytorch_lightning" in path.read_text()]

    assert stale == []


def test_graph_data_module_uses_lightning2_base():
    from hamgnn.data.graph_data import graph_data_module

    assert graph_data_module.__bases__[0].__module__.startswith("lightning.pytorch")
