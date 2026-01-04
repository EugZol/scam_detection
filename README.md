# Scam Detection

This project implements a machine learning pipeline for detecting scam messages (scam detection) using text classification. The pipeline includes data preprocessing, model training, and inference capabilities.

## Screenshots



## Message classification

The goal of this project is to develop an automated system that can classify messages as either "Safe Message" or "Scam Message" based on their textual content. This is achieved through natural language processing techniques and machine learning models.

Key features:

- Text preprocessing and feature extraction
- Traditional ML (TF-IDF + Logistic Regression) for baseline and deep learning (Transformer-based) model
- Model export to ONNX format
- Model serving via MLflow
- Experiment tracking with MLflow
- Data version control with DVC

## Technical Details

### Setup

1. **Prerequisites:**
   - Python 3.9+
   - Git
   - uv package manager
   - MLflow server, running at 127.0.0.1:8080 (configurable)

2. **Clone the repository:**
   ```bash
   git clone git@github.com:EugZol/scam_detection.git
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
   uv run mlflow server --host 127.0.0.1 --port 8080
   ```

### Train

Train the model using the main command interface:

```bash
# Default: Small transformer model
uv run python -m scam_detection.commands train

# TF-IDF baseline logistic regression
uv run python -m scam_detection.commands train model=baseline

# Override training parameters
uv run python -m scam_detection.commands train train.max_epochs=10 data.batch_size=32
```

**Available models:**
- `model=small_transformer` - Transformer-based model (default)
- `model=baseline` - TF-IDF + Logistic Regression

**Note:** Training automatically pulls data from DVC.

The training process includes data preprocessing, model training with PyTorch Lightning, validation with early stopping, metrics logging to MLflow, and checkpoint saving to `models/checkpoints/`.

### Production preparation

**Export model to ONNX format:**

Only available for transformer model, not baseline.

```bash
# Interactive: select default checkpoint
uv run python -m scam_detection.commands export

# Specify checkpoint and output paths
uv run python -m scam_detection.commands export -m models/checkpoints/best.ckpt -o models/onnx/model.onnx
```

The exported ONNX model be deployed in production environments.

### Infer

Run inference on new messages using trained models:
- **Transformer model**: Uses PyTorch checkpoint files (`.ckpt`) from `models/checkpoints/`
- **TF-IDF model**: Uses MLflow registered model

```bash
# Inference with automatic checkpoint selection (transformer)
uv run python -m scam_detection.commands infer \
  infer.texts='["Congratulations! You won a prize!", "Meeting at 3pm"]'

# Inference with specific checkpoint (transformer)
uv run python -m scam_detection.commands infer \
  infer.model_path=models/checkpoints/best.ckpt \
  infer.texts='["Congratulations! You won a prize!", "Meeting at 3pm"]'

# Inference from file (one message per line)
uv run python -m scam_detection.commands infer \
  infer.input_file=readme_assets/messages.txt

# Inference with baseline model (uses MLflow)
uv run python -m scam_detection.commands infer model=baseline \
  infer.texts='["Test message"]'
```

**Input format:**
- `infer.texts`: List of strings in Python syntax
- `infer.input_file`: Text file with one message per line

**Output:** Predictions printed with labels (SCAM/SAFE) for each message.

### Serve

Deploy the model using MLflow serving:

```bash
# Serve latest registered small_transformer model
MLFLOW_TRACKING_URI=http://127.0.0.1:8080 uv run mlflow models serve \
  -m models:/small_transformer/latest \
  -p 8000 \
  --env-manager local

# Serve baseline TF-IDF model
MLFLOW_TRACKING_URI=http://127.0.0.1:8080 uv run mlflow models serve \
  -m models:/tfidf/latest \
  -p 8000 \
  --env-manager local

# Or use a specific run
MLFLOW_TRACKING_URI=http://127.0.0.1:8080 uv run mlflow models serve \
  -m runs:/<RUN_ID>/model \
  -p 8000 \
  --env-manager local
```

**Notes:**
- `MLFLOW_TRACKING_URI` points to the tracking server where models are registered
- `--env-manager local` uses the current environment instead of creating a new one

**Test the served model:**
```bash
curl -X POST http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Your message text here"]}'
```

The model URI can be found in the MLflow UI after training at `http://127.0.0.1:8080`.

### Test suite

```bash
uv run pytest
```

Test suite includes some smoke tests and CLI interface tests.

## Data preparation

Data is a CSV file with columns:
- `Message Text`: String containing the message content
- `Message Type`: Classification label ("Safe Message" or "Scam Message")

Data is managed with DVC and downloaded from a remote source. During training, data is automatically pulled if not present locally. To manually download or update the data file, run:

```bash
dvc pull
```
