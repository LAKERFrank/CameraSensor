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
