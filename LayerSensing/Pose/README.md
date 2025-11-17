# TensorRT Pose Inference Requirements

The pose worker uses a TensorRT engine (`*.engine`) and requires NVIDIA runtime libraries in the Python environment.
If TensorRT bindings are missing, startup will fail with an error similar to:

```
RuntimeError: TensorRT Python bindings are required to use the quantized pose engine. Install them with `pip install tensorrt==10.7.0` (and ensure compatible CUDA/cuDNN runtime libraries are present).
```

To resolve this:

1. Install TensorRT Python bindings (matching your engine build). For the current pose engine use:
   ```bash
   pip install tensorrt==10.7.0
   ```
2. Install the CUDA runtime Python bindings (cuda-python):
   ```bash
   pip install cuda-python
   ```
3. Install CUDA/cuDNN shared libraries so TensorRT can load `libcublas.so.11` and `libcudnn_ops_infer.so.8` (or their matching versions). Options:
   * Using NVIDIA wheels (works in many container setups):
     ```bash
     pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
     ```
   * Or install system packages that ship these libraries (e.g., CUDA Toolkit/cuDNN runtime matching the TensorRT version).
4. Ensure a supported NVIDIA driver/GPU is available. Without GPU support, the quantized TensorRT pose engine cannot run.
5. Verify the engine file path passed to `startPose` exists. You can pass just the engine filename if the file is placed under `LayerSensing/Pose/engine/<engine_name>`; otherwise, provide an absolute path.

After installing the dependencies, rerun the pose worker (e.g., `python3 main_device.py ...`) and the error should no longer appear.

## Engine deserialization errors (e.g., `magicTag` mismatch)

If you see TensorRT messages such as `Serialization assertion magicTagRead == kMAGIC_TAG failed` or
`deserializeCudaEngine: Error Code 4: Internal Error`, the engine file is either corrupted or built
with a different TensorRT version than the runtime on this device. To fix:

1. Verify the engine file size is reasonable (tens of MB) and matches what you exported.
2. Re-export the model using the **same TensorRT version** as the runtime (check the runtime version
   in the logs when loading the engine). For Ultralytics YOLOv8, a typical command is:
   ```bash
   yolo export model=yolov8n-pose.pt format=engine device=0 half=True int8=True
   ```
3. Copy the regenerated engine to `LayerSensing/Pose/engine/` (or pass the absolute path to
   `startPose`).
4. Restart the pose worker.

## Troubleshooting missing CUDA/cuDNN libraries

If startup fails with errors like `Could not load library libcudnn_ops_infer.so.8` or `libcublas.so.11: cannot open shared object file`, install the CUDA/cuDNN runtime that matches the TensorRT version used to export the engine (10.7.0). Typical fixes include installing the NVIDIA runtime wheels (`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`) or the corresponding system packages for your CUDA Toolkit version, then restarting the pose worker.
