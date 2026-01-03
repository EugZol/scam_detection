# Scam Detection Project

This project implements a machine learning system for detecting scams, specifically focusing on scam message detection. The project must be packaged as a valid Python package on GitHub, implementing data loading, preprocessing, training, production preparation, and inference capabilities.

## Project Requirements Overview

To receive full credit, the project must satisfy all mandatory conditions listed below. Submissions not meeting these requirements will receive 0 points.

### Mandatory Conditions
- Repository accessible via provided GitHub link
- Code version submitted in the main branch (master or main)
- Project topic matches the specified scam detection assignment
- All main sections (marked with *) are implemented in at least basic form

## Implementation Steps

### 1. Repository and Package Setup
Create an open GitHub repository with a valid Python package structure. The repository name should reflect the project meaning (e.g., scam-detection) using dashes as separators, not underscores.

### 2. Environment and Dependencies Management (5 points) *
Choose either Poetry or uv for dependency management. Dependencies must be specified in `pyproject.toml` with corresponding lock file (`poetry.lock` or `uv.lock`). Ensure no extra unused dependencies are included.

### 3. Code Quality Tools (10 points) *
Implement pre-commit with the following hooks:
- pre-commit
- black
- isort
- flake8
- prettier (for non-Python files)

Running `pre-commit run -a` must pass without errors.

### 4. Training Framework (5 points) *
Use PyTorch Lightning as the basic framework (other discussed frameworks acceptable but with fewer capabilities). Training must leverage framework features rather than custom implementations.

### 5. Data Management (10 points) *
Use DVC for data storage with one of:
- Google Drive
- S3 (ensure accessibility)
- Local storage (implement `download_data()` function for open-source data)

Integrate DVC data downloading into train and infer commands using Python API or CLI.

### 6. Configuration Management with Hydra (10 points) *
Convert hyperparameters for preprocessing, training, and postprocessing to Hydra YAML configs in a `configs/` folder. Group configs by operations and use hierarchical structure. Remove magic constants from code.

### 7. Logging (5 points) *
Implement logging of at least 3 main metrics and loss functions. Log hyperparameters and git commit ID. Assume MLflow server running at 127.0.0.1:8080. Store graphs and logs in `plots/` directory. May use additional systems like WandB.

### 8. README.md Documentation (10 points) *
Create comprehensive documentation with two main sections:

#### Semantic Content
Copy and adapt project description from previous assignment, maintaining proper formatting and structure.

#### Technical Details
- **Setup**: Environment configuration procedure using chosen dependency manager
- **Train**: Commands for running training (including data loading, preprocessing, model options)
- **Production Preparation**: Steps for model preparation (ONNX, TensorRT conversion) and delivery artifacts
- **Infer**: Commands for model inference on new data, including data format and examples

### 9. Model Production Packaging (10 points)
- **ONNX Conversion (5 points)**: Convert model to ONNX format, optionally including preprocessing/postprocessing
- **TensorRT Conversion (5 points)**: Convert from ONNX or directly from model using shell/Python CLI. Add example data to DVC

### 10. Inference Server (10 points)
Implement using either:
- **MLflow Serving (5 points max)**: Setup and usage instructions
- **Triton Inference Server (10 points max)**: Setup and usage instructions

### 11. Validation Steps
The project will be validated through:
1. Repository cloning
2. Virtual environment creation per instructions
3. Dependency installation
4. `pre-commit install` and `pre-commit run -a` success
5. Training execution
6. Prediction system execution

## Code Standards and Best Practices

### Prohibited Actions (-3 points each)
- No executable code at file level (use `if __name__ == '__main__':`)
- No top-level variable declarations (except constants)
- No `warnings.filterwarnings("ignore")`
- No data/model files in Git (.json, .csv, .h5, .cbm, .pth, .onnx, etc.)
- Repository/package names must follow Python conventions (snake_case) and reflect project meaning
- Use snake_case for code files, not CamelCase
- No argparse; use fire, click, hydra, or similar
- No single-letter variables (except i, j, k); use semantic names

### Recommended Practices (+2 points)
- Call one main function under `if __name__ == '__main__':`
- Use fire with hydra via compose API
- Single entry point `commands.py` calling functions from other files
- Use pathlib instead of os.path

## Checklist Flags
1. Use "messages" terminology, not "emails"
2. Use proper logging instead of printf
3. Ensure simple transformer appears in MLflow models tab
4. Prevent runs from being tracked as failed in MLflow

## Additional Notes
This specification provides guidelines rather than strict dogmas. Adapt techniques to the task while following course knowledge and good practices. Contact instructors for clarification on unclear points or alternative approaches.
