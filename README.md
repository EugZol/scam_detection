# Scam Detection

This project implements a machine learning pipeline for detecting scam emails (phishing detection) using text classification. The pipeline includes data preprocessing, model training, and inference capabilities.

## Semantic Content

The goal of this project is to develop an automated system that can classify emails as either "Safe Email" or "Phishing Email" based on their textual content. This is achieved through natural language processing techniques and machine learning models.

Key features:
- Text preprocessing and feature extraction
- Support for both traditional ML (TF-IDF + Logistic Regression) and deep learning (Transformer-based) approaches
- Production-ready model export to ONNX and TensorRT formats
- Model serving via MLflow or Triton Inference Server
- Experiment tracking with MLflow
- Data version control with DVC

## Technical Details

### Setup

1. **Prerequisites:**
   - Python 3.9+
   - Git
   - uv package manager

2. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd scam_detection
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

5. **Run pre-commit on all files:**
   ```bash
   uv run pre-commit run -a
   ```

6. **Start MLflow server (in separate terminal):**
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

### Train

To train the model:

1. **For Transformer model:**
   ```bash
   uv run scam_detection
   ```
   This uses the default config (transformer model).

2. **For TF-IDF baseline:**
   Modify `configs/config.yaml` to use `model: baseline.yaml`, then run the same command.

The training process includes:
- Data loading and preprocessing
- Model training with PyTorch Lightning
- Validation and early stopping
- Logging metrics to MLflow
- Saving checkpoints

### Production Preparation

1. **Export to ONNX:**
   ```bash
   uv run python scripts/export_onnx.py models/checkpoints/best.ckpt models/onnx/model.onnx distilbert-base-uncased
   ```

The exported ONNX model is optimized for inference and can be deployed in production environments.

### Infer

To run inference:

1. **Using the trained model:**
   ```bash
   uv run python scripts/smoke_test_infer.py
   ```

2. **Serving with MLflow:**
   - The model is logged to MLflow during training
   - Use MLflow's serving capabilities to deploy the model

Input format: JSON with "text" field containing the email content.
Output: Prediction score (0 for safe, 1 for phishing).

## Dependencies

- PyTorch (CPU version)
- PyTorch Lightning
- Transformers
- Scikit-learn
- MLflow
- DVC
- Hydra
- And others (see pyproject.toml)

## Project Structure

```
scam_detection/
├── data/                 # Data loading and preprocessing
├── models/               # Model definitions
├── training/             # Training scripts
├── inference/            # Inference components
├── serving/              # Model serving
├── configs/              # Hydra configurations
├── scripts/              # Utility scripts
└── tests/                # Unit tests
```

## Contributing

1. Follow the coding standards enforced by pre-commit hooks
2. Write tests for new features
3. Update documentation as needed

## License

[Add license information]
