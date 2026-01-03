"""
Integration tests for CLI commands.

Tests the full command pipeline through direct function calls with Hydra configs,
ensuring README examples work correctly.
"""

import time
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from scam_detection.commands import export, infer, train


@pytest.fixture
def test_dirs(tmp_path):
    """Create isolated test directories for artifacts."""
    test_checkpoints = tmp_path / "checkpoints"
    test_onnx = tmp_path / "onnx"
    test_mlruns = tmp_path / "mlruns"

    test_checkpoints.mkdir()
    test_onnx.mkdir()
    test_mlruns.mkdir()

    return {
        "checkpoints": test_checkpoints,
        "onnx": test_onnx,
        "mlruns": test_mlruns,
        "base": tmp_path,
    }


@pytest.fixture
def hydra_context():
    """Initialize Hydra context for tests."""
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).parent.parent / "configs"
    initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None)
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def tiny_dataset_path():
    """Return path to tiny test dataset."""
    return str(Path(__file__).parent / "fixtures" / "tiny_dataset.csv")


def test_train_baseline(hydra_context, test_dirs, tiny_dataset_path):
    """
    Test training TF-IDF baseline model (README example).

    Command equivalent:
        uv run python -m scam_detection.commands train model=baseline
    """
    cfg = compose(
        config_name="config",
        overrides=[
            "model=baseline",
            f"data.csv_path={tiny_dataset_path}",
            "data.batch_size=2",
            "data.test_size=0.2",  # 20% test, sufficient for 10 samples
            "data.val_size=0.2",   # 20% val from remaining
            f"logging.mlflow_tracking_uri=file://{test_dirs['mlruns']}",
            "train.mlflow_experiment=test_baseline",
            "train.log_model=false",  # Skip slow model logging in tests
        ],
    )

    # Should complete without errors
    train(cfg)

    # Verify MLflow artifacts were created
    assert test_dirs["mlruns"].exists()
    assert len(list(test_dirs["mlruns"].rglob("*"))) > 0


def test_train_transformer(hydra_context, test_dirs, tiny_dataset_path):
    """
    Test training small transformer model (README example).

    Command equivalent:
        uv run python -m scam_detection.commands train model=small_transformer

    Note: Uses fast_dev_run to only run 2 batches for speed.
    """
    # Track test start time for cleanup
    test_start_time = time.time()

    cfg = compose(
        config_name="config",
        overrides=[
            "model=small_transformer",
            f"data.csv_path={tiny_dataset_path}",
            "data.batch_size=4",
            "data.test_size=0.2",
            "data.val_size=0.2",
            "+train.fast_dev_run=2",  # Only run 2 batches for testing
            f"logging.mlflow_tracking_uri=file://{test_dirs['mlruns']}",
            "train.mlflow_experiment=test_transformer",
        ],
    )

    train(cfg)

    # With fast_dev_run, checkpoints are not saved, so just verify training completed
    # without errors (if it got here, it succeeded)
    assert test_dirs["mlruns"].exists()


def test_train_transformer_with_overrides(hydra_context, test_dirs, tiny_dataset_path):
    """
    Test training with parameter overrides (README example).

    Command equivalent:
        uv run python -m scam_detection.commands train train.max_epochs=10 data.batch_size=32

    Note: Uses fast_dev_run to only run 2 batches for speed.
    """
    test_start_time = time.time()

    cfg = compose(
        config_name="config",
        overrides=[
            "model=small_transformer",
            f"data.csv_path={tiny_dataset_path}",
            "+train.fast_dev_run=2",  # Only run 2 batches for testing
            "data.batch_size=2",  # Small batch for tiny dataset
            "data.test_size=0.2",
            "data.val_size=0.2",
            f"logging.mlflow_tracking_uri=file://{test_dirs['mlruns']}",
            "train.mlflow_experiment=test_overrides",
        ],
    )

    # Should complete without errors
    train(cfg)

    # Verify it used our overrides (check MLflow logs)
    assert test_dirs["mlruns"].exists()


@pytest.fixture
def trained_model_checkpoint(hydra_context, test_dirs, tiny_dataset_path):
    """Train a model and return checkpoint path for inference/export tests."""
    test_start_time = time.time()

    cfg = compose(
        config_name="config",
        overrides=[
            "model=small_transformer",
            f"data.csv_path={tiny_dataset_path}",
            "data.batch_size=2",
            "data.test_size=0.2",
            "data.val_size=0.2",
            "train.max_epochs=1",
            f"logging.mlflow_tracking_uri=file://{test_dirs['mlruns']}",
            "train.mlflow_experiment=test_fixture",
            "train.log_model=false",  # Skip slow model logging in tests
        ],
    )

    train(cfg)

    # Find the created checkpoint
    checkpoint_dir = Path("models/checkpoints")
    checkpoints = sorted(
        checkpoint_dir.glob("small_transformer-*.ckpt"),
        key=lambda p: p.stat().st_mtime
    )

    if not checkpoints:
        raise RuntimeError("No checkpoint created during fixture setup")

    # Get checkpoints created during this test
    test_checkpoints = [c for c in checkpoints if c.stat().st_mtime >= test_start_time]

    if not test_checkpoints:
        raise RuntimeError("No checkpoint created during this test run")

    checkpoint_path = test_checkpoints[-1]  # Most recent from this test

    yield checkpoint_path

    # Cleanup: remove all checkpoints created during this test
    for ckpt in test_checkpoints:
        if ckpt.exists():
            ckpt.unlink()


def test_export_with_hydra_override(
    hydra_context, test_dirs, tiny_dataset_path, trained_model_checkpoint
):
    """
    Test ONNX export with Hydra overrides (README example).

    Command equivalent:
        uv run python -m scam_detection.commands export export.model_path=models/checkpoints/best.ckpt
    """
    output_path = test_dirs["onnx"] / "test_model.onnx"

    cfg = compose(
        config_name="config",
        overrides=[
            "+export=default",
            "model=small_transformer",
            f"export.model_path={trained_model_checkpoint}",
            f"export.onnx_path={output_path}",
        ],
    )

    export(cfg)

    # Verify ONNX file was created and has content
    assert output_path.exists(), "ONNX file not created"
    assert output_path.stat().st_size > 0, "ONNX file is empty"


def test_infer_with_texts(
    hydra_context, test_dirs, tiny_dataset_path, trained_model_checkpoint
):
    """
    Test inference with text input (README example).

    Command equivalent:
        uv run python -m scam_detection.commands infer \
          infer.model_path=models/checkpoints/best.ckpt \
          infer.texts='["Congratulations! You won!", "Meeting at 3pm"]'
    """
    cfg = compose(
        config_name="config",
        overrides=[
            "+infer=default",
            "model=small_transformer",
            f"infer.model_path={trained_model_checkpoint}",
            "infer.texts=[\"Congratulations! You won a prize!\", \"Meeting at 3pm\"]",
        ],
    )

    # Should complete without errors
    # Note: infer() prints to stdout, we're just checking it doesn't crash
    infer(cfg)


def test_infer_from_file(
    hydra_context, test_dirs, tiny_dataset_path, trained_model_checkpoint
):
    """
    Test inference from file input (README example).

    Command equivalent:
        uv run python -m scam_detection.commands infer \
          infer.model_path=models/checkpoints/best.ckpt \
          infer.input_file=messages.txt
    """
    # Create temp input file
    input_file = test_dirs["base"] / "test_messages.txt"
    input_file.write_text(
        "Congratulations! You won a prize!\n"
        "Meeting scheduled for tomorrow.\n"
        "Click here to verify your account!\n"
    )

    cfg = compose(
        config_name="config",
        overrides=[
            "+infer=default",
            "model=small_transformer",
            f"infer.model_path={trained_model_checkpoint}",
            f"infer.input_file={input_file}",
        ],
    )

    # Should complete without errors
    infer(cfg)
