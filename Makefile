PWD=$(shell pwd)
DOCKER_GID=$(shell getent group docker | cut -d: -f3)

.PHONY: build-image push-image pull-image run-device-monitor-daemon run-camera run dev-local dev-remote

build-image: Dockerfile tis111.dockerfile
	docker build -t gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/camera-sensor:v1.1 .
	docker build -t gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/camera-sensor:tis_111 --file tis111.dockerfile .
	docker build -t gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/app:latest --file app.dockerfile .

push-image:
	docker push gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/camera-sensor:v1.1
	docker build -t gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/app:latest --file app.dockerfile .

pull-image:
	docker pull gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/camera-sensor:v1.1

run-device-monitor-daemon:
	docker run -it \
		--privileged \
		--gpus all \
		--shm-size=8g \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		-e PROJECT_HOST_DIR=${PWD} \
		-e PYTHONPATH=/app \
		-v ./:/app/ \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v /run/udev:/run/udev \
		-w /app \
		--group-add ${DOCKER_GID}\
		--add-host=host.docker.internal:host-gateway \
		--name "device-monitor" \
		--restart unless-stopped \
		--detach \
		gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/camera-sensor:tis_111 \
		python3 LayerCamera/device_monitor.py

run-camera:
	docker run --rm -it \
		--privileged \
		--gpus all \
		--shm-size=8g \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		-e DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./:/app/ \
		-w /app/ \
		--add-host=host.docker.internal:host-gateway \
		gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/camera-sensor:v1.1 \
		python3 main_app.py local

run:
	docker run --rm -it \
		-e DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./:/app/ \
		-w /app \
		--add-host=host.docker.internal:host-gateway \
		gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/app:latest \
		python3 main_app.py

dev-local:
	rm -rf .devcontainer
	mkdir .devcontainer
	cp scripts/devcontainer_local.json .devcontainer/devcontainer.json

dev-remote:
	rm -rf .devcontainer
	mkdir .devcontainer
	cp scripts/devcontainer_remote.json .devcontainer/devcontainer.json
