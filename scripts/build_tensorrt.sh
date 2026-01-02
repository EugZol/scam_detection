#!/bin/bash
"""
Build TensorRT engine from ONNX model.
"""

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <onnx_path> <trt_path>"
    exit 1
fi

ONNX_PATH=$1
TRT_PATH=$2

# Use trtexec to build engine
trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_PATH --fp16

echo "TensorRT engine built at $TRT_PATH"
