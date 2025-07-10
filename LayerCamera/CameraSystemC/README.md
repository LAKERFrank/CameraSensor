# CameraSystemC
* high performance camera recorder
* camera: Imaging Source
    * 37BUX252, 37AUX273
* Put under NOL_Playground to read related configs

## Requirements
* OS: Ubuntu22.04⬆️
    * If your OS is Ubuntu 20.04, you must manually install some GStreamer plugins and g++-11.
    * If your OS is Ubuntu 22.04, after completing the installation of GStreamer Bad Plugins, the following elements will be included.
* Required GStreamer plugins
    * You can use the `gst-inspect-1.0` command to check the installed plugins.
    * `nvh264enc`, `cudaupload`, `cudascale`, `cudaconvert`
* `nvh264enc`
    * An element of NVENC
    * Can be built by ourself
* `cudaupload`, `cudascale`, `cudaconvert`
    * Elements of CUDA
    * I haven't found a method for manual installation yet.
    * My Makefile will detect whether `cudascale` exists. If it does not exist, it will automatically use `videoscale` and `videoconvert` (CPU).

## Compile stand alone UI
```
qmake
make
```

## Compile widget for python
```
make -f Makefile_widget
```
