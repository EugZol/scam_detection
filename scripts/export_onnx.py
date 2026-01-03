#!/usr/bin/env python3
"""
Export model to ONNX format.
"""

import torch
from transformers import AutoTokenizer

from scam_detection.models.lit_module import MessageClassifier


def export_to_onnx(
    model_path: str,
    onnx_path: str,
    tokenizer_name: str = "bert-base-uncased",
    model_type: str = "small_transformer",
):
    """
    Export PyTorch model to ONNX.

    Args:
        model_path: Path to PyTorch model
        onnx_path: Path to save ONNX model
        tokenizer_name: Tokenizer name
    """
    # Load model
    model = MessageClassifier.load_from_checkpoint(
        model_path, model_type=model_type, tokenizer_name=tokenizer_name
    )

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

    model.eval()

    # Export to ONNX
    torch.onnx.export(
        model.model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )

    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in {4, 5}:
        print(
            "Usage: python export_onnx.py <model_path> <onnx_path> <tokenizer_name> "
            "[model_type]"
        )
        sys.exit(1)

    model_type = sys.argv[4] if len(sys.argv) == 5 else "small_transformer"
    export_to_onnx(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        model_type=model_type,
    )
