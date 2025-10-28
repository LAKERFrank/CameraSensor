ARG CUDA_VERSION=12.4.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=nol
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV TZ=Asia/Taipei

# Install gstreamer
RUN sed -i 's/archive.ubuntu.com/free.nchc.org.tw/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-gl gstreamer1.0-gtk3 \
        gstreamer1.0-qt5 libgstrtspserver-1.0-dev

RUN apt-get install -y gstreamer1.0-plugins-base-apps

# enable "source"
SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && dpkg-reconfigure locales \
    && update-locale LANG=en_US.UTF-8

ENV LC_ALL=en_US.UTF-8

# Create the user
RUN apt-get update \
    && apt-get install -y sudo \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --groups sudo,video -m -s /bin/bash $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install common tools
RUN apt-get install -y git vim curl wget usbutils lsb-release pkg-config

# Build tiscamera
RUN cd /tmp \
    && git clone https://github.com/TheImagingSource/tiscamera.git \
    && cd tiscamera \
    && git checkout v-tiscamera-1.0.0 \
    && sed -i 's/tags\[\] = {"id", "val", NULL}/tags\[\] = {NULL}/g' \
        src/gstreamer-1.0/tcamsrc/gstmetatcamstatistics.cpp \
    && sudo ./scripts/dependency-manager install -y \
    && mkdir build \
    && cd build \
    # Sphinx would produce error, so disabled TCAM_BUILD_DOCUMENTATION
    && cmake -DTCAM_BUILD_DOCUMENTATION=OFF -DTCAM_BUILD_ARAVIS=OFF .. \
    && make -j \
    && sudo make install \
    && rm -rf /tmp/tiscamera

# environment packages
RUN apt-get install -y freeglut3 freeglut3-dev mosquitto mosquitto-clients \
    libxml2 libxml2-dev libusb-1.0-0-dev libzip-dev qtbase5-dev qtdeclarative5-dev libudev-dev \
    python3-sphinx python3-pyqt5 libpulse-dev libturbojpeg libcap-dev \
    build-essential lsb-release cmake pkg-config

# for opencv?
RUN apt-get install -y ffmpeg gfortran liblapack-dev liblapacke-dev libopenblas-dev libeigen3-dev libtbb2 libtbb-dev \
    libv4l-dev v4l-utils qv4l2 gpac x264 libx264-dev libgtk-3-dev python3-vtk7

USER ${USERNAME}

# Install minconda3
RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.9.0-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh \
    && ~/miniconda3/bin/conda init bash

ENV PATH="/home/${USERNAME}/miniconda3/bin/:$PATH"

COPY requirements.txt /home/${USERNAME}/requirements.txt

RUN pip3 install -r ~/requirements.txt

# Fix: ImportError: /home/nol/miniconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /workspaces/camerasensor/LayerCamera/CameraSystemC/recorder_module.cpython-38-x86_64-linux-gnu.so)
RUN conda install -c conda-forge -y libstdcxx-ng=12
