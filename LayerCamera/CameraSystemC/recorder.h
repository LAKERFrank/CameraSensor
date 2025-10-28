// recorder.h
#ifndef RECORDER_H
#define RECORDER_H

#include <gst/gst.h>
#include <gst/video/videooverlay.h>

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <variant>
#include <filesystem>
#include <latch>
#include <map>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <string>
#include <variant>

#include "common.h"
#include "modules/loop_record.h"
#include "modules/record.h"
#include "tcamcamera.h"

namespace fs = std::filesystem;

class Recorder {
 public:
  Recorder(std::string root_dir);
  ~Recorder();
  void init(std::string cam_serial, unsigned long win_id,
            std::string netstream_host, int netstream_port, int direction, bool clockoverlay, bool software_trigger, bool resync);
  void start(unsigned long long trigger_start);
  void release();
  void startRecording(std::string save_path, bool imgbuf,
                      int imgbuf_width, int imgbuf_height, std::string mode);
  void stopRecording();
  void startLoopRecording();
  void stopLoopRecording();

  void enableDisplay(unsigned long win_id);
  void disableDisplay();

  void enableUdp(std::string host, int port);
  void disableUdp();

  std::shared_ptr<gsttcam::Snapshot> takeSnapshot();

  std::vector<std::map<std::string, std::string>> listAvailableCamera();

  std::shared_ptr<gsttcam::ImageBuffer> getImageBuffer();
  std::shared_ptr<gsttcam::MetricData> getMetricData();

  std::vector<std::map<std::string, std::variant<int, std::string, std::vector<int>>>> getCaptureFormats();

  std::map<std::string, std::string> getDeviceInfo();

  void setProperty(std::string name, std::string value);
  void setCaptureFormat(int width, int height, int fps, std::string skipping="");

  double startVideoFeeder(std::string video_path, bool enable_imgbug);
  void stopVideoFeeder();

  bool isStreaming() const { return _isStreaming; }

  void resync(unsigned long long t, unsigned long long dt);

 private:

  void startLoopRecordingCam();
  void stopLoopRecordingCam();

  void runTestVideo(std::string video_path, bool enable_imgbuf = true);

 private:
  fs::path ROOTDIR;
  std::map<std::string, std::map<std::string, std::string>> main_config;
  std::map<std::string, std::map<std::string, std::string>> camera_config;

  std::shared_ptr<gsttcam::TcamCamera> cam;

  gsttcam::CameraInfo cam_device_info;

  std::vector<gsttcam::CameraInfo> cam_devices_info_all;

  bool _isStreaming = false;
  bool isRecording = false;
  bool isLoopRecording = false;

  std::shared_ptr<gsttcam::ImageBuffer> imgbuf;
  std::shared_ptr<gsttcam::MetricData> metric_data;

  std::thread videoFeederThread;
};

#endif  // RECORDER_H