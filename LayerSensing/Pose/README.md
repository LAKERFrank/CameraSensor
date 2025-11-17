# TensorRT Pose Inference Requirements

The pose worker uses a TensorRT engine (`*.engine`) and requires NVIDIA runtime libraries in the Python environment.
If TensorRT bindings are missing, startup will fail with an error similar to:

```
RuntimeError: TensorRT Python bindings are required to use the quantized pose engine. Install them via `pip install --index-url https://pypi.ngc.nvidia.com nvidia-tensorrt`.
```

To resolve this:

1. Install TensorRT Python bindings from NVIDIA's index:
   ```bash
   pip install --index-url https://pypi.ngc.nvidia.com nvidia-tensorrt
   ```
2. Install the CUDA runtime Python bindings (cuda-python):
   ```bash
   pip install cuda-python
   ```
3. Ensure a supported NVIDIA driver/GPU is available. Without GPU support, the quantized TensorRT pose engine cannot run.
4. Verify the engine file path passed to `startPose` exists. You can pass just the engine filename if the file is placed under `LayerSensing/Pose/engine/<engine_name>`; otherwise, provide an absolute path.

After installing the dependencies, rerun the pose worker (e.g., `python3 main_device.py ...`) and the error should no longer appear.
