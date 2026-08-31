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


def test_user_docs_describe_lightning2_resume_and_keep_num_gpus():
    text = "\n".join(
        path.read_text()
        for path in [Path("README.md"), *Path("docs/source/user_guide").glob("*.rst")]
    )
    assert "lightning>=2.5,<2.6" in text or "Lightning 2" in text
    assert "trainer.fit" in text and "ckpt_path" in text
    assert "num_gpus" in text
    assert "examples/V1.0" not in text


def test_historical_examples_are_removed():
    assert not Path("examples/V1.0").exists()


def test_sphinx_intersphinx_uses_lightning_namespace():
    assert "lightning.pytorch" in Path("docs/source/conf.py").read_text()
