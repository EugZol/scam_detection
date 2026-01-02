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
   - Docker and docker-compose
   - Git

2. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd scam_detection
   ```

3. **Build and start services:**
   ```bash
   docker-compose up -d
   ```
   
   Note: The first build will take time to install dependencies. Subsequent runs will be much faster as dependencies are cached in a Docker volume.

4. **Install dependencies (first time only):**
   ```bash
   docker-compose exec scam_detection uv sync
   ```

5. **Install pre-commit hooks (inside container):**
   ```bash
   docker-compose exec scam_detection uv run pre-commit install
   ```

6. **Run pre-commit on all files:**
   ```bash
   docker-compose exec scam_detection uv run pre-commit run -a
   ```

### Train

To train the model:

1. **For Transformer model:**
   ```bash
   docker-compose exec scam_detection uv run scam_detection
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
   docker-compose exec scam_detection python scripts/export_onnx.py models/checkpoints/best.ckpt models/onnx/model.onnx distilbert-base-uncased
   ```

2. **Build TensorRT engine:**
   ```bash
   docker-compose exec scam_detection bash scripts/build_tensorrt.sh models/onnx/model.onnx models/trt/model.trt
   ```

The exported models are optimized for inference and can be deployed in production environments.

### Infer

To run inference:

1. **Using the trained model:**
   ```bash
   docker-compose exec scam_detection python scripts/smoke_test_infer.py
   ```

2. **Serving with MLflow:**
   - The model is logged to MLflow during training
   - Use MLflow's serving capabilities to deploy the model

3. **Serving with Triton:**
   - Convert the ONNX model to Triton format
   - Deploy using Triton Inference Server

Input format: JSON with "text" field containing the email content.
Output: Prediction score (0 for safe, 1 for phishing).

## Dependencies

- PyTorch (with ROCm support for AMD GPUs)
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
