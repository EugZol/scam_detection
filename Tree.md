scam_detection/
├─ .gitignore
├─ .pre-commit-config.yaml
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ dvc.yaml
├─ dvc.lock
├─ .dvc/
│  ├─ config
│  └─ .gitignore
├─ data/
│  ├─ raw/.gitkeep
│  ├─ interim/.gitkeep
│  └─ processed/.gitkeep
├─ models/
│  ├─ checkpoints/.gitkeep
│  ├─ onnx/.gitkeep
│  └─ trt/.gitkeep
├─ artifacts/
│  ├─ mlflow/.gitkeep
│  └─ examples/.gitkeep
├─ plots/
│  ├─ train/.gitkeep
│  └─ eval/.gitkeep
├─ scripts/
│  ├─ export_onnx.py
│  ├─ build_tensorrt.sh
│  └─ smoke_test_infer.py
├─ configs/
│  ├─ config.yaml
│  ├─ data.yaml
│  ├─ preprocess.yaml
│  ├─ train.yaml
│  ├─ eval.yaml
│  ├─ infer.yaml
│  ├─ export.yaml
│  ├─ serve.yaml
│  ├─ logging/
│  │  ├─ default.yaml
│  │  └─ minimal.yaml
│  ├─ hydra/
│  │  ├─ default.yaml
│  │  └─ job_logging.yaml
│  └─ model/
│     ├─ baseline.yaml
│     └─ transformer.yaml
├─ scam_detection/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ commands.py
│  ├─ constants.py
│  ├─ types.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ seed.py
│  │  ├─ io.py
│  │  ├─ logging.py
│  │  └─ hashing.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ download.py
│  │  ├─ schema.py
│  │  ├─ preprocessing.py
│  │  ├─ dataset.py
│  │  └─ datamodule.py
│  ├─ features/
│  │  ├─ __init__.py
│  │  ├─ tokenizer.py
│  │  └─ vectorizer.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ lit_module.py
│  │  ├─ baseline.py
│  │  ├─ transformer.py
│  │  └─ registry.py
│  ├─ training/
│  │  ├─ __init__.py
│  │  ├─ trainer.py
│  │  ├─ callbacks.py
│  │  └─ metrics.py
│  ├─ evaluation/
│  │  ├─ __init__.py
│  │  ├─ evaluator.py
│  │  └─ reports.py
│  ├─ inference/
│  │  ├─ __init__.py
│  │  ├─ predictor.py
│  │  ├─ onnx_runtime.py
│  │  ├─ trt_runtime.py
│  │  └─ postprocess.py
│  ├─ serving/
│  │  ├─ __init__.py
│  │  ├─ mlflow_model.py
│  │  ├─ fastapi_app.py
│  │  └─ schemas.py
│  └─ tracking/
│     ├─ __init__.py
│     ├─ mlflow.py
│     └─ git_info.py
└─ tests/
   ├─ __init__.py
   ├─ conftest.py
   ├─ test_preprocess.py
   ├─ test_train_smoke.py
   ├─ test_infer_smoke.py
   └─ test_export_smoke.py
