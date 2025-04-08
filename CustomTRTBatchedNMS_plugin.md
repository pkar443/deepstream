# DeepStream Pipeline with Custom TRTBatchedNMS Plugin - Setup & Usage Guide

## Overview
This document outlines how to set up and run a DeepStream 6.4 pipeline using a **custom-built TRTBatchedNMS plugin** from MMDeploy. The plugin enables optimized postprocessing using TensorRT and removes the need for `.so`-based custom parsers.

---

## Prerequisites
- Docker with GPU support
- NVIDIA GPU Drivers + CUDA + cuDNN
- DeepStream 6.4 base image
- MMDeploy source code available inside the container
- YOLO model exported as TensorRT engine

---

##  1. Build Custom TRTBatchedNMS Plugin
Run the following **inside your DeepStream Docker container**:

```bash
# Step 1: Clean & Create build folder
cd /workspace/mmdeploy
rm -rf build
mkdir build && cd build

# Step 2: Prepare fake TensorRT SDK (if needed)
mkdir -p /workspace/tensorrt_fake/include
cp /usr/include/x86_64-linux-gnu/NvInfer*.h /workspace/tensorrt_fake/include
mkdir -p /workspace/tensorrt_fake/lib
cp /usr/lib/x86_64-linux-gnu/libnvinfer* /workspace/tensorrt_fake/lib

# Step 3: Run CMake
cmake .. \
  -DMMDEPLOY_TARGET_BACKENDS=trt \
  -DTENSORRT_DIR=/workspace/tensorrt_fake \
  -DMMDEPLOY_BUILD_SDK=OFF \
  -DMMDEPLOY_BUILD_SDK_PYTHON_API=OFF

# Step 4: Build Plugin
make -j$(nproc)
```

> The plugin will be generated at:
```
/workspace/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so
```

---

## 2. Add Plugin to DeepStream Runtime
Instead of using the deprecated `plugin-library` key in config, preload the plugin before running DeepStream:

```bash
export LD_PRELOAD=/workspace/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so

```

---


