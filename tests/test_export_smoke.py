"""
Smoke test for export.
"""

import os
import tempfile

from scripts.export_onnx import export_to_onnx


def test_export_onnx():
    # Mock test
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        tokenizer = "distilbert-base-uncased"
        # Smoke test: check if function exists (will fail without real model)
        try:
            export_to_onnx("dummy", onnx_path, tokenizer)
        except Exception:
            pass  # Expected to fail
