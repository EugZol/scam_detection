from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from .data.datamodule import MessageDataModule
from .inference.predictor import Predictor
from .models.lit_module import MessageClassifier
from .training.trainer import train_tfidf_model, train_transformer_model


def train(cfg: DictConfig):
    """
    Train the model.

    Args:
        cfg: Hydra config
    """
    # Pull data using DVC if not available locally
    import subprocess
    from pathlib import Path

    data_path = Path(cfg.data.csv_path)
    if not data_path.exists():
        print(f"Data file {data_path} not found locally. Pulling from DVC...")
        try:
            # Use DVC CLI to pull the data
            result = subprocess.run(
                ["uv", "run", "dvc", "pull"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            if result.returncode == 0:
                print("Successfully pulled data from DVC")
            else:
                print(f"Failed to pull data from DVC: {result.stderr}")
                print("Please ensure DVC is properly configured and data is available")
                return
        except Exception as e:
            print(f"Failed to run DVC pull: {e}")
            print("Please ensure DVC is properly configured and data is available")
            return

    # Prepare datamodule kwargs based on model type
    datamodule_kwargs = {
        "csv_path": cfg.data.csv_path,
        "model_type": cfg.model.model_type,
        "batch_size": cfg.data.batch_size,
        "test_size": cfg.data.test_size,
        "val_size": cfg.data.val_size,
        "random_state": cfg.data.random_state,
        "num_workers": cfg.data.num_workers,
    }

    # Add transformer-specific parameters if using transformer model
    if cfg.model.model_type in {"transformer", "small_transformer"}:
        datamodule_kwargs["tokenizer_name"] = cfg.model.tokenizer_name
        datamodule_kwargs["max_length"] = cfg.model.max_length

    datamodule = MessageDataModule(**datamodule_kwargs)

    if cfg.model.model_type in {"transformer", "small_transformer"}:
        # NOTE: we keep the name `train_transformer_model` for compatibility,
        # but it now trains the scratch-based small transformer.
        train_transformer_model(
            datamodule=datamodule,
            model_config=cfg.model,
            learning_rate=cfg.train.learning_rate,
            max_epochs=cfg.train.max_epochs,
            patience=cfg.train.patience,
            mlflow_experiment=cfg.train.mlflow_experiment,
            mlflow_tracking_uri=cfg.logging.mlflow_tracking_uri,
            log_every_n_steps=cfg.logging.log_every_n_steps,
            cpu_threads=cfg.train.cpu_threads,
        )
    elif cfg.model.model_type == "tfidf":
        train_tfidf_model(
            datamodule,
            cfg.train.mlflow_experiment,
            cfg.logging.mlflow_tracking_uri,
        )


def infer(cfg: DictConfig):
    """
    Run inference on input texts.

    Args:
        cfg: Hydra config
    """
    import ast

    # Check for required parameters
    if "model_path" not in cfg.infer:
        print("Error: model_path is required for inference")
        print(
            "Example: python -m scam_detection.commands infer "
            "infer.model_path=path/to/model.ckpt"
        )
        return

    model_path = Path(cfg.infer.model_path)
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found")
        return

    # Get model type and tokenizer from config
    model_type = cfg.model.model_type
    tokenizer_name = cfg.model.get("tokenizer_name", "distilbert-base-uncased")

    # Load model
    print(f"Loading model from {model_path}...")
    model = MessageClassifier.load_from_checkpoint(
        str(model_path),
        model_type=model_type,
        tokenizer_name=tokenizer_name,
    )

    # Load tokenizer if needed
    tokenizer = None
    if model_type in {"transformer", "small_transformer"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    predictor = Predictor(model, tokenizer)

    # Get input texts
    if "texts" in cfg.infer:
        # Parse texts from command line
        texts_str = cfg.infer.texts
        if isinstance(texts_str, str):
            try:
                texts = ast.literal_eval(texts_str)
            except Exception:
                texts = [texts_str]
        else:
            texts = list(texts_str)
    elif "input_file" in cfg.infer:
        # Read texts from file
        input_file = Path(cfg.infer.input_file)
        if not input_file.exists():
            print(f"Error: Input file {input_file} not found")
            return
        with open(input_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Either texts or input_file must be provided")
        print(
            "Example: python -m scam_detection.commands infer "
            "infer.model_path=path/to/model.ckpt "
            "infer.texts='[\"Test message\"]'"
        )
        return

    # Run inference
    print(f"Running inference on {len(texts)} texts...")
    predictions = predictor.predict(texts)

    # Display results
    print("\nResults:")
    for text, pred in zip(texts, predictions):
        label = "SCAM" if pred == 1 else "SAFE"
        print(f"  [{label}] {text}")


def export(cfg: DictConfig):
    """
    Export model to ONNX format.

    Args:
        cfg: Hydra config
    """
    # Check for required parameters
    if "model_path" not in cfg.export:
        print("Error: model_path is required for export")
        print(
            "Example: python -m scam_detection.commands export "
            "export.model_path=path/to/model.ckpt"
        )
        return

    model_path = Path(cfg.export.model_path)
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found")
        return

    onnx_path = Path(cfg.export.get("onnx_path", "models/onnx/model.onnx"))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model_type = cfg.model.model_type
    tokenizer_name = cfg.model.get("tokenizer_name", "distilbert-base-uncased")
    max_length = cfg.model.get("max_length", 512)

    # Load model
    print(f"Loading model from {model_path}...")
    model = MessageClassifier.load_from_checkpoint(
        str(model_path),
        model_type=model_type,
        tokenizer_name=tokenizer_name,
    )

    if model_type not in {"transformer", "small_transformer"}:
        print(
            "Error: ONNX export only supported for transformer models, "
            f"not {model_type}"
        )
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create dummy input
    input_sample = cfg.export.get("input_sample", "This is a test message.")
    print(f"Creating dummy input: '{input_sample}'")
    dummy_input = tokenizer(
        input_sample,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    model.eval()

    # Export to ONNX
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model.model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )

    print(f"Successfully exported model to {onnx_path}")


def serve(cfg: DictConfig):
    """
    Start MLflow serving.

    Args:
        cfg: Hydra config
    """
    print("MLflow serving is not yet fully implemented.")
    print("To serve a model with MLflow:")
    print("1. Ensure your model is logged to MLflow")
    print("2. Run: mlflow models serve -m <model_uri> -p 8000")
    print("\nConfiguration:")
    print(f"  Host: {cfg.serve.get('host', '0.0.0.0')}")
    print(f"  Port: {cfg.serve.get('port', 8000)}")
    print(f"  Model Path: {cfg.serve.get('model_path', 'N/A')}")


def main():
    """
    Main CLI entry point.
    """
    import sys

    # Extract command from first argument
    if len(sys.argv) < 2:
        print("Usage: python -m scam_detection.commands <command> [options]")
        print("\nAvailable commands:")
        print("  train   - Train a model")
        print("  infer   - Run inference on texts")
        print("  export  - Export model to ONNX format")
        print("  serve   - Start MLflow serving")
        print("\nExample:")
        print("  python -m scam_detection.commands train model=baseline")
        return

    command = sys.argv[1]

    # Check if command is a valid command or a Hydra override
    valid_commands = ["train", "infer", "export", "serve"]
    if command not in valid_commands and "=" not in command:
        print(f"Error: Unknown command '{command}'")
        print(f"Valid commands: {', '.join(valid_commands)}")
        print("Or use Hydra overrides like: model=baseline")
        return

    # If no command or command is a Hydra override, default to train
    if command not in valid_commands:
        command = "train"
        overrides = sys.argv[1:]
    else:
        overrides = sys.argv[2:]

    # Initialize Hydra
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).parent.parent / "configs"
    initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None)

    # Add default config groups for different commands
    if command == "infer":
        default_overrides = ["+infer=default"]
    elif command == "export":
        default_overrides = ["+export=default"]
    elif command == "serve":
        default_overrides = ["+serve=default"]
    else:
        default_overrides = []

    cfg = compose(config_name="config", overrides=default_overrides + overrides)
    print(OmegaConf.to_yaml(cfg))

    # Route to appropriate command
    if command == "train":
        train(cfg)
    elif command == "infer":
        infer(cfg)
    elif command == "export":
        export(cfg)
    elif command == "serve":
        serve(cfg)


if __name__ == "__main__":
    # Don't use Fire here - we want to handle args ourselves for Hydra
    main()
