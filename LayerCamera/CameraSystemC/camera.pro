# camera.pro

# Specify the gcc and g++ versions
QMAKE_CC  = gcc-11
QMAKE_CXX = g++-11

# Avoid using -isystem to include paths, as it may lead to include_next errors.
QMAKE_CFLAGS_ISYSTEM =

# Target name
TARGET = camera

# Tell qmake that this project requires the Qt Widgets module
QT += widgets

# Check if the cudascale element exists
CUDASCALE_EXISTS = $$system(gst-inspect-1.0 cudascale > /dev/null 2>&1 && echo 1 || echo 0)
DEFINES += CUDASCALE_EXISTS=$${CUDASCALE_EXISTS}

# Tell qmake that this project requires the GStreamer and glib-2.0 modules
CONFIG += link_pkgconfig
PKGCONFIG += gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0 glib-2.0 tcam opencv4
QMAKE_CXXFLAGS += -std=c++20
QMAKE_LFLAGS +=

# Source files
SOURCES += main.cpp mainwindow.cpp tcamcamera.cpp recorder.cpp common.cpp modules/record.cpp modules/snapshot.cpp modules/display.cpp modules/udp.cpp modules/loop_record.cpp

# Header files (if any)
HEADERS += mainwindow.h tcamcamera.h recorder.h common.h modules/record.h modules/snapshot.h modules/display.h modules/udp.h modules/loop_record.h


