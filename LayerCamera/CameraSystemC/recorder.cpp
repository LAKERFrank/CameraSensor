#include "recorder.h"


namespace fs = std::filesystem;

std::map<std::string, std::map<std::string, std::string>> readConfig(
    const std::string& filename) {
  std::map<std::string, std::map<std::string, std::string>> config;
  std::ifstream file(filename);
  std::string line;
  std::string currentSection;

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }

  while (std::getline(file, line)) {
    // Trim whitespace from the beginning and end of the line
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);

    if (!line.empty() && line[0] == '[' && line[line.size() - 1] == ']') {
      // It's a section header
      currentSection = line.substr(1, line.size() - 2);
    } else if (!line.empty() && line.find('=') != std::string::npos) {
      // It's a key-value pair
      std::istringstream is_line(line);
      std::string key, value;
      getline(is_line, key, '=');
      getline(is_line, value);
      // Trim whitespaces
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);
      // LOGMYSYSTEM << "[" << key << "], [" << value << "]\n";
      config[currentSection][key] = value;
    }
  }

  file.close();

  return config;
}

std::pair<int, int> parsePairFromString(const std::string& str) {
  std::pair<int, int> result;
  char ignore;  // Used to ignore characters like '(', ',', and ')'
  std::istringstream iss(str);

  iss >> ignore;         // Ignore the '('
  iss >> result.first;   // Parse the first integer
  iss >> ignore;         // Ignore the ','
  iss >> result.second;  // Parse the second integer

  return result;
}

Recorder::Recorder(std::string root_dir) {
  LOGMYSYSTEM << "[recorder constructor]" << "\n";
  ROOTDIR = root_dir;
  ROOTDIR = fs::absolute(ROOTDIR);
  LOGMYSYSTEM << "rootdir = " << ROOTDIR << "\n";
  gst_init(nullptr, nullptr);

  imgbuf = std::make_shared<gsttcam::ImageBuffer>();
  metric_data = std::make_shared<gsttcam::MetricData>();

  return;
}

Recorder::~Recorder() {
  LOGMYSYSTEM << "[recorder destructor]" << "\n";
  this->release();
  return;
}

std::vector<std::map<std::string, std::string>>
Recorder::listAvailableCamera() {
  cam_devices_info_all = gsttcam::get_device_list();

  std::vector<std::map<std::string, std::string>> ret;
  for (gsttcam::CameraInfo cam_info : cam_devices_info_all) {
    std::map<std::string, std::string> item = {
        {"name", cam_info.name},
        {"serial", cam_info.serial},
    };
    // LOGMYSYSTEM << "[list camera]" << cam_info.serial << std::endl;

    ret.push_back(item);
  }
  return ret;
}

void Recorder::init(std::string cam_serial, unsigned long win_id,
                    std::string netstream_host, int netstream_port,
                    int direction, bool clockoverlay, bool software_trigger, bool resync) {
  LOGMYSYSTEM << "[recorder init]" << "\n";

  cam_devices_info_all = gsttcam::get_device_list();

  LOGMYSYSTEM << "(USB) get device list ok num=" << cam_devices_info_all.size()
              << "\n";

  for (gsttcam::CameraInfo cam_info : cam_devices_info_all) {
    LOGMYSYSTEM << "(USB) exist -> " << cam_info.serial << " " << cam_info.name
                << " " << cam_info.identifier << " " << cam_info.connection_type
                << "\n";
  }

  // =========================================================

  bool found = false;

  // find camera device info
  for (unsigned long j = 0; j < cam_devices_info_all.size(); j++) {
    if (cam_devices_info_all[j].serial == cam_serial) {
      cam_device_info = cam_devices_info_all[j];
      found = true;
      break;
    }
  }

  if (!found && cam_serial != std::string("None")) {
    throw std::runtime_error("No device available serial=" + cam_serial);
  }

  // =========================================================

  LOGMYSYSTEM << "open camera\n";

  cam = std::make_shared<gsttcam::TcamCamera>(cam_serial, imgbuf, metric_data, clockoverlay, software_trigger, resync);
  cam->set_direction(direction);
  
  LOGMYSYSTEM << "open camera..ok\n";

  // =========================================================

  // preview setup
  LOGMYSYSTEM << "start preview setup\n";

  if (win_id != 0) {
    cam->enable_video_display(win_id);
  }

  if (netstream_host != "" && netstream_port != 0) {
    cam->enable_video_udp_stream(netstream_host, netstream_port);
  }

  cam->enable_video_snapshot();
  cam->enable_video_record();
  cam->enable_loop_record();

  LOGMYSYSTEM << "preview setup...ok\n";
  // =========================================================

  isRecording = false;
}

void Recorder::start(unsigned long long trigger_start) {
  // Show the live vieo
  cam->start(trigger_start);
  _isStreaming = true;
}

void Recorder::release() {
  LOGMYSYSTEM << "[recorder release]" << "\n";
  if (cam != nullptr) {
    LOGMYSYSTEM << "releasing\n";
    cam->stop();
    cam = nullptr;
    _isStreaming = false;
  }
  LOGMYSYSTEM << "recorder release...ok\n";
}

void Recorder::enableUdp(std::string host, int port) {
  cam->enable_video_udp_stream(host, port);
}

void Recorder::disableUdp() {
  cam->disable_video_udp_stream();
}

void Recorder::enableDisplay(unsigned long win_id) {
  cam->enable_video_display(win_id);
}
void Recorder::disableDisplay() {
  cam->disable_video_display();
}

void Recorder::startRecording(std::string save_path, bool imgbuf, 
                              int imgbuf_width, int imgbuf_height, std::string mode) {
  LOGMYSYSTEM << "[recorder start recording]" << "\n";

  if (isRecording) {
    LOGMYSYSTEM << "!!(Warning) The recorder is currently recording. !!\n";
    return;
  }

  fs::path filepath = save_path;

  if (!filepath.has_filename()) {
    throw std::runtime_error("filepath does not has a filename");
  }

  fs::path recorddir = filepath;
  recorddir.remove_filename();

  try {
    if (!fs::create_directories(recorddir)) {
      LOGMYSYSTEM << "Directory exists." << std::endl;
    }
  } catch (const fs::filesystem_error& e) {
    LOGMYSYSTEM << "Filesystem error: " << e.what() << std::endl;
  }

  LOGMYSYSTEM << "video saving to " << filepath.string() << std::endl;

  cam->start_recording(filepath.string(), imgbuf, imgbuf_width, imgbuf_height, mode);

  //fs::copy(ROOTDIR / "Reader" / "Image_Source" / "config" /
  //             (cam->getSerial() + ".cfg"),
  //         recorddir / (cam->getSerial() + ".cfg"));

  isRecording = true;
};

void Recorder::stopRecording() {
  LOGMYSYSTEM << "[recorder stop recording]" << "\n";

  if (!isRecording) {
    LOGMYSYSTEM << "!!(Warning) The recorder is not recording. !!\n";
    return;
  }
  cam->stop_recording();
  isRecording = false;
  return;
};

void Recorder::startLoopRecording() {
  LOGMYSYSTEM << "[recorder start loop recording]" << "\n";

  if (isRecording) {
    LOGMYSYSTEM << "!!(Warning) The recorder is currently loop recording. !!\n";
    return;
  }

  auto now = std::chrono::system_clock::now();

  // Convert to time_t for formatting
  auto now_c = std::chrono::system_clock::to_time_t(now);

  // Create a stringstream to hold the formatted time
  std::stringstream ss;

  // Format the time and output to the stringstream
  // You can customize the format string as needed
  ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S");

  // Convert the stringstream to string
  std::string timestamp = ss.str();

  startLoopRecordingCam();

  // timer->start();
  // startTime = QTime::currentTime();
  isLoopRecording = true;

  return;
};

void Recorder::stopLoopRecording() {
  LOGMYSYSTEM << "[recorder stop loop recording]" << "\n";

  if (!isLoopRecording) {
    LOGMYSYSTEM << "!!(Warning) The recorder is not loop recording. !!\n";
    return;
  }
  stopLoopRecordingCam();
  isLoopRecording = false;
  return;
};

void Recorder::setProperty(std::string name, std::string value) {
  if (this->cam) {
    this->cam->setProperty(name, value);
  }
}

std::shared_ptr<gsttcam::Snapshot> Recorder::takeSnapshot() {
  if (cam_device_info.serial == "None") {
    return nullptr;
  }
  return cam->take_snapshot();
}

std::shared_ptr<gsttcam::ImageBuffer> Recorder::getImageBuffer() {
  return imgbuf;
}

std::shared_ptr<gsttcam::MetricData> Recorder::getMetricData() {
  return metric_data;
}

void Recorder::startLoopRecordingCam() {
  LOGMYSYSTEM << "[start loop recording cam]" << "\n";
  cam->startLoopRecording();
};

void Recorder::stopLoopRecordingCam() { cam->stopLoopRecording(); };

std::vector<std::map<std::string, std::variant<int, std::string, std::vector<int>>>> Recorder::getCaptureFormats() {
  std::vector<std::map<std::string, std::variant<int, std::string, std::vector<int>>>> res;
  if (cam_device_info.name == "DFK 37BUX252") {
    res.push_back(std::map<std::string, std::variant<int, std::string, std::vector<int>>>
      {{"width", 2048}, {"height", 1536}, {"skipping", "1x1"}, {"target_fps", std::vector<int> {119, 60, 30}}});
    res.push_back(std::map<std::string, std::variant<int, std::string, std::vector<int>>>
      {{"width", 1024}, {"height", 768}, {"skipping", "2x2"}, {"target_fps", std::vector<int> {238, 120, 60, 30}}});
  } else if (cam_device_info.name == "DFK 37AUX273") {
    res.push_back(std::map<std::string, std::variant<int, std::string, std::vector<int>>>
      {{"width", 1440}, {"height", 1080}, {"skipping", "1x1"}, {"target_fps", std::vector<int> {236, 120, 60, 30}}});
    res.push_back(std::map<std::string, std::variant<int, std::string, std::vector<int>>>
      {{"width", 640}, {"height", 480}, {"skipping", "2x2"}, {"target_fps", std::vector<int> {600, 480, 240, 120, 60, 30}}});
  }
  return res;
}

std::map<std::string, std::string> Recorder::getDeviceInfo() {
  return std::map<std::string, std::string>{
      {"brand", "Image_Source"},
      {"serial", cam_device_info.serial},
      {"model", cam_device_info.name},
  };
}

void Recorder::runTestVideo(std::string video_path, bool enable_imgbuf) {
  cv::VideoCapture cap(video_path);

  if (!cap.isOpened()) {
    throw std::runtime_error("!!! Failed to open file");
  }

  cv::Mat img;
  std::shared_ptr<gsttcam::Frame> frame;
  int i = 0;

  struct timespec ts1, ts2;

  clock_gettime(CLOCK_MONOTONIC, &ts1);

  for (;;) {
    if (!cap.read(img)) {
      break;
    }

    // control playback speed
    double video_pos = cap.get(cv::CAP_PROP_POS_MSEC);
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    double real_pos = (ts2.tv_sec * 1e3 + ts2.tv_nsec / 1e6) -
                      (ts1.tv_sec * 1e3 + ts1.tv_nsec / 1e6);
    if (video_pos > real_pos) {
      usleep(int((video_pos - real_pos) * 1e3));
    }

    cv::resize(img, img, cv::Size(512, 288));
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    cv::Size sz = img.size();

    if (enable_imgbuf) {
      frame = std::make_shared<gsttcam::Frame>();
      frame->index = i++;
      frame->timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0f;
      frame->monotonic_timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0f;
      frame->width = sz.width;
      frame->height = sz.height;
      frame->memory_size = img.step[0] * img.rows;
      std::shared_ptr<u_int8_t> memory = std::shared_ptr<u_int8_t>(
          (u_int8_t*)malloc(frame->memory_size), free);
      memcpy(memory.get(), img.data, frame->memory_size);
      frame->memory = memory;
      frame->is_eos = false;

      imgbuf->push(frame);
    }
  }

  cap.release();

  frame = std::make_shared<gsttcam::Frame>();
  frame->index = i++;
  frame->is_eos = true;

  imgbuf->push(frame);
}

double Recorder::startVideoFeeder(std::string video_path, bool enable_imgbuf)
{
  // stop existing feeder
  this->stopVideoFeeder();

  cv::VideoCapture cap(video_path);
  double fps = cap.get(cv::CAP_PROP_FPS);
  double frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
  double duration = frame_count / fps;
  cap.release();

  this->videoFeederThread = std::thread(&Recorder::runTestVideo, this, video_path, enable_imgbuf);

  return duration;
}

void Recorder::stopVideoFeeder()
{
  if (this->videoFeederThread.joinable()) {
    this->videoFeederThread.join();
  }
}

void Recorder::setCaptureFormat(int width, int height, int target_fps, std::string skipping) {
  if (this->cam) {
    gsttcam::FrameRate framerate {target_fps, 1};
    // override framerate
    if (cam_device_info.name == "DFK 37AUX273") {
      if (width == 1440 && height == 1080 && target_fps == 236) {
        framerate = {2500000, 10593};
      }
      else if (width == 640 && height == 480 && skipping == "2x2" && target_fps == 240) {
        framerate = {5000000, 20833};
      }
    }
    else if (cam_device_info.name == "DFK 37BUX252") {
      if (width == 1024 && height == 768 && skipping == "2x2" && target_fps == 238) {
        framerate = {312500, 1313};
      }
    }
    this->cam->set_capture_format(gsttcam::FrameSize{width, height}, framerate, skipping);
  }
}

void Recorder::resync(unsigned long long t, unsigned long long dt) {
  if (this->cam) {
    this->cam->resync(t, dt);
  }
}

#ifdef BUILD_WIDGET
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

PYBIND11_MODULE(recorder_module, m) {
  py::class_<gsttcam::Snapshot, std::shared_ptr<gsttcam::Snapshot>>(m,
                                                                    "Snapshot")
      .def(py::init<>())
      .def_property_readonly("width", &gsttcam::Snapshot::getWidth)
      .def_property_readonly("height", &gsttcam::Snapshot::getHeight)
      .def_property_readonly(
          "image",
          [](gsttcam::Snapshot& s) {
            return py::array_t<uint8_t>(py::buffer_info(
                s.memory.get(), sizeof(uint8_t),              // itemsize
                py::format_descriptor<uint8_t>::format(), 3,  // ndim
                std::vector<size_t>{s.height, s.width, 3},    // shape
                std::vector<size_t>{s.width * sizeof(uint8_t) * 3,
                                    sizeof(uint8_t) * 3, sizeof(uint8_t)}
                // stride
                ));
          },
          "numpy.ndarray of BGR image");
  py::class_<gsttcam::Frame, std::shared_ptr<gsttcam::Frame>>(m, "Frame")
      .def(py::init<>())
      .def_property_readonly("width", &gsttcam::Frame::getWidth)
      .def_property_readonly("height", &gsttcam::Frame::getHeight)
      .def_property_readonly("index", &gsttcam::Frame::getIndex)
      .def_property_readonly("timestamp", &gsttcam::Frame::getTimestamp)
      .def_property_readonly("monotonic_timestamp", &gsttcam::Frame::getMonotonicTimestamp)
      .def_property("is_eos", &gsttcam::Frame::getIsEOS, &gsttcam::Frame::setIsEOS , "Is end of stream")
      .def_property_readonly("image", [](gsttcam::Frame& f) {
        return py::array_t<uint8_t> (py::buffer_info(f.memory.get(), sizeof(uint8_t), // itemsize
                                                    py::format_descriptor<uint8_t>::format(), 2, // ndim
                                                    std::vector<size_t> {(unsigned long)f.height, (unsigned long)f.width}, // shape
                                                    std::vector<size_t> {f.width * sizeof(uint8_t), sizeof(uint8_t)} // stride
                                                  ));
      }, "numpy.ndarray of BGR image");
  py::class_<gsttcam::ImageBuffer, std::shared_ptr<gsttcam::ImageBuffer>>(m, "ImageBuffer")
      .def(py::init<>())
      .def(
          "pop",
          [](gsttcam::ImageBuffer& imgbuf, bool blocking = true) {
            py::gil_scoped_release release;
            return imgbuf.pop(blocking);
          },
          "", "blocking"_a = true)
      .def("push", &gsttcam::ImageBuffer::push)
      .def("clear", &gsttcam::ImageBuffer::clear);

  py::class_<gsttcam::MetricData, std::shared_ptr<gsttcam::MetricData>>(m, "MetricData")
      .def_property_readonly("fps", [](gsttcam::MetricData& m) { return m.fps; })
      .def_property_readonly("avg_fps", [](gsttcam::MetricData& m) { return m.avg_fps; })
      .def_property_readonly("frames_rendered", [](gsttcam::MetricData& m) { return m.frames_rendered; })
      .def_property_readonly("frames_dropped", [](gsttcam::MetricData& m) { return m.frames_dropped; })
      .def_property_readonly("kf", [](gsttcam::MetricData& m) {
        double timestamp, dt;
        m.get_kf(timestamp, dt);
        return std::vector<double> {timestamp, dt};
      })
      .def("serialize", [](gsttcam::MetricData& m) {
        double timestamp, dt;
        m.get_kf(timestamp, dt);
        py::dict d;
        d["fps"] = m.fps;
        d["avg_fps"] = m.avg_fps;
        d["frames_rendered"] = m.frames_rendered;
        d["frames_dropped"] = m.frames_dropped;
        d["kf_timestamp"] = timestamp;
        d["kf_dt"] = dt;
        return d;
      });

  py::class_<Recorder>(m, "Recorder")
      .def(py::init<const std::string&>())
      .def("init", &Recorder::init, "", "cam_serial"_a, "win_id"_a,
           "udp_host"_a, "udp_port"_a, "direction"_a, "clockoverlay"_a=false, "software_trigger"_a=false, "resync"_a=false)
      .def("start", &Recorder::start, "", "trigger_start"_a=NULL)
      .def("getDeviceInfo", &Recorder::getDeviceInfo)
      .def("getCaptureFormats", &Recorder::getCaptureFormats)
      .def("setCaptureFormat", &Recorder::setCaptureFormat, "", "width"_a, "height"_a, "target_fps"_a, "skipping"_a="")
      .def("release", &Recorder::release)
      .def("enableDisplay", &Recorder::enableDisplay, "", "win_id"_a)
      .def("disableDisplay", &Recorder::disableDisplay)
      .def("enableUdp", &Recorder::enableUdp, "", "host"_a, "port"_a)
      .def("disableUdp", &Recorder::disableUdp)
      .def("startRecording", &Recorder::startRecording, "", "save_path"_a, "imgbuf"_a, "imgbuf_width"_a, "imgbuf_height"_a, "mode"_a)
      .def("stopRecording", &Recorder::stopRecording)
      .def("listAvailableCamera", &Recorder::listAvailableCamera)
      .def("takeSnapshot", &Recorder::takeSnapshot, "take single snapshot")
      .def("getImageBuffer", &Recorder::getImageBuffer)
      .def("getMetricData", &Recorder::getMetricData)
      .def("startVideoFeeder", &Recorder::startVideoFeeder, "", "video_path"_a, "enable_imgbuf"_a)
      .def("stopVideoFeeder", &Recorder::stopVideoFeeder, "")
      .def("setProperty", &Recorder::setProperty, "", "name"_a, "value"_a)
      .def("resync", &Recorder::resync,
        "Synchronize camera with reference timestamp.\n\n"
        "Args:\n"
        "  resync_timestamp_ns (int): Timestamp in nanoseconds. Must not be in the future.",
        "resync_timestamp_ns"_a, "resync_dt_ns"_a)
      .def_property_readonly("isStreaming", &Recorder::isStreaming, "Is camera streaming.");
}
#endif
