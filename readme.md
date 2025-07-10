# Camera Sensor

## Usage

* Main APP

```
make run
```

* CES APP

```
make run-ces
```

* Camera App

```
make run-camera
```

* MQTT Device Monitor

```bash
make run-device-monitor-daemon
```

## Notice

1. 有插新相機的話需要重開
2. 如果有重設相機位置的話也需要重開

## RpcCamera

### run server

```sh
python LayerCamera/rpc/server.py
```

### test client

```
python LayerCamera/camera/RpcCamera.py
```

## devcontainer setup

* Develop on remote

	```sh
	make dev-remote
	```

* Develop on local

	```sh
	make dev-local
	```
* Install qt5 designer (Ubuntu)

	```sh
	sudo apt-get install qttools5-dev-tools
	sudo apt-get install qttools5-dev
	```

## Dockerfile history

* v1.2
	* base image from `nvcr.io/nvidia/pytorch:23.08-py3` to `nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04`
	* apt source using `free.nchc.org.tw` for faster speed
	* add psutil to `requirements.txt`
* v1.3
	* add pyqtgraph to `requirements.txt`