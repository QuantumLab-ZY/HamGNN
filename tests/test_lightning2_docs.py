from pathlib import Path

import yaml


def test_dependency_files_declare_lightning_25():
    environment = yaml.safe_load(Path("HamGNN.yaml").read_text())
    assert "lightning>=2.5,<2.6" in environment["dependencies"]
    docs_environment = yaml.safe_load(Path("docs/environment.yml").read_text())
    assert "lightning>=2.5,<2.6" in docs_environment["dependencies"]
    assert all("pytorch-lightning" not in str(item) for item in environment["dependencies"])
    assert all("pytorch-lightning" not in str(item) for item in docs_environment["dependencies"])


def test_setup_metadata_requires_supported_python_and_lightning():
    setup_text = Path("setup.py").read_text()
    assert 'python_requires=">=3.9"' in setup_text
    assert '"lightning>=2.5,<2.6"' in setup_text


def test_documented_pytorch_pyg_stack_matches_environment():
    environment = yaml.safe_load(Path("docs/environment.yml").read_text())
    assert "pytorch=2.5.0" in environment["dependencies"]
    pip_dependencies = environment["dependencies"][-1]["pip"]
    assert "--find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html" in pip_dependencies
    assert "torch-geometric==2.6.1" in pip_dependencies

    text = "\n".join(
        path.read_text()
        for path in [Path("README.md"), Path("docs/source/user_guide/installation.rst")]
    )
    assert "PyTorch == 2.5.0" in text
    assert "PyTorch Geometric == 2.6.1" in text
    assert "torch-2.5.0+cu121.html" in text


def test_user_docs_describe_lightning2_resume_and_keep_num_gpus():
    text = "\n".join(
        path.read_text()
        for path in [Path("README.md"), *Path("docs/source/user_guide").glob("*.rst")]
    )
    assert "lightning>=2.5,<2.6" in text or "Lightning 2" in text
    assert "trainer.fit(model, datamodule, ckpt_path=checkpoint_path)" in text
    assert "resume: true" in text and "checkpoint_path" in text
    assert "num_gpus: null" in text and "positive integer" in text
    assert "null" in text and "cpu" in text
    assert all(value in text for value in ("'cpu'", "'gpu'", "'ddp'"))
    assert "precision: 32" in text and "64" in text
    assert "examples/V1.0" not in text


def test_documented_checkpoint_path_default_matches_runtime():
    from hamgnn.config.config_parsing import config_default

    parameters = Path("docs/source/user_guide/parameters.rst").read_text()
    assert "``checkpoint_path``" in parameters
    assert "- ``'./'``" in parameters
    assert config_default["setup"]["checkpoint_path"] == "./"


def test_v2_example_exposes_migrated_setup_contract():
    setup = yaml.safe_load(Path("examples/V2.x/config.yaml").read_text())["setup"]
    assert setup["num_gpus"] == 1
    assert setup["precision"] == 32
    assert setup["accelerator"] is None
    assert setup["resume"] is False
    assert setup["checkpoint_path"] is None


def test_current_references_exclude_legacy_checkpoint_metadata():
    current_files = [
        Path("HamGNN.yaml"),
        Path("docs/environment.yml"),
        Path("setup.py"),
        Path("README.md"),
        *Path("docs/source").rglob("*.rst"),
    ]
    legacy_import = "pytorch" + "_lightning"
    assert all(legacy_import not in path.read_text() for path in current_files)

    # This key is intentional metadata written by old checkpoints, not a current import or dependency.
    legacy_version_key = "pytorch" + "-lightning_version"
    assert legacy_version_key in Path("tests/test_lightning2_model.py").read_text()


def test_historical_examples_are_removed():
    assert not Path("examples/V1.0").exists()


def test_sphinx_intersphinx_uses_lightning_namespace():
    assert "lightning.pytorch" in Path("docs/source/conf.py").read_text()
