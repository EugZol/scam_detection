#!/usr/bin/env python3
"""
Export model to ONNX format.
"""

import torch
from transformers import AutoTokenizer

from scam_detection.models.lit_module import EmailClassifier


def export_to_onnx(
    model_path: str, onnx_path: str, tokenizer_name: str = "distilbert-base-uncased"
):
    """
    Export PyTorch model to ONNX.

    Args:
        model_path: Path to PyTorch model
        onnx_path: Path to save ONNX model
        tokenizer_name: Tokenizer name
    """
    # Load model
    model = EmailClassifier.load_from_checkpoint(model_path, model_type="transformer")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create dummy input
    dummy_input = tokenizer(
        "This is a test email.",
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    # Export to ONNX
    torch.onnx.export(
        model.model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=11,
    )

    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python export_onnx.py <model_path> <onnx_path> <tokenizer_name>")
        sys.exit(1)

    export_to_onnx(sys.argv[1], sys.argv[2], sys.argv[3])
