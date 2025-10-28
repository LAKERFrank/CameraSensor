#ifndef __TCAMCAMERA_H__
#define __TCAMCAMERA_H__

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>

#include <condition_variable>
#include <fstream>
#include <functional>
#include <latch>
#include <memory>
#include <mutex>
#include <opencv4/opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "common.h"
#include "modules/display.h"
#include "modules/record.h"
#include "modules/snapshot.h"
#include "modules/udp.h"
#include "modules/loop_record.h"
#include "modules/monitor.h"
#include "tcam-property-1.0.h"

#define LOGMYSYSTEM std::cout << "[CameraSystemC] "

namespace gsttcam {

class TcamCamera;

/*
 *
 */
struct CameraInfo {
  std::string serial;
  std::string name;
  std::string identifier;
  std::string connection_type;
};

/*
 *
 */
struct FrameRate {
  int numerator;
  int denominator;
};

/*
 *
 */
struct FrameSize {
  int width;
  int height;
};

/*
 *
 */
class VideoFormatCaps {
 public:
  FrameSize size;
  FrameSize size_min;
  FrameSize size_max;
  std::vector<std::string> formats;
  std::vector<FrameRate> framerates;
  FrameRate framerate_min;
  FrameRate framerate_max;

  std::string to_string();
};

class Property {
 public:
  TcamPropertyBase *baseProperty;
  std::string name;
  std::string displayName;
  std::string category;
  std::string description;
  std::string type;
  std::string unit;

  bool available;
  bool locked;

  Property(TcamPropertyBase *baseProperty, std::string name);
  ~Property();

  virtual std::string to_string() {};
};

class IntegerProperty : public Property {
 public:
  TcamPropertyInteger *integerProperty;
  long int value;
  long int default_value;
  long int min;
  long int max;
  long int step;

  IntegerProperty(TcamPropertyBase *base, std::string name);

  virtual std::string to_string() override { return ""; };
  bool set(int value);
};

class DoubleProperty : public Property {
 public:
  TcamPropertyFloat *doubleProperty;
  double value;
  double default_value;
  double min;
  double max;
  double step;

  DoubleProperty(TcamPropertyBase *base, std::string name);

  virtual std::string to_string() override { return ""; };
  bool set(double value);
};

class EnumProperty : public Property {
 public:
  TcamPropertyEnumeration *enumProperty;

  std::string value;
  std::string default_value;

  std::vector<std::string> values;

  EnumProperty(TcamPropertyBase *base, std::string name);

  virtual std::string to_string() override { return ""; };
  bool set(std::string value);
};

class BooleanProperty : public Property {
 public:
  TcamPropertyBoolean *booleanProperty;
  bool value;
  bool default_value;

  BooleanProperty(TcamPropertyBase *base, std::string name);

  virtual std::string to_string() override { return ""; };
  bool set(bool value);
};

class TcamCamera {
 public:
  TcamCamera(std::string serial, std::shared_ptr<ImageBuffer> image_queue,
             std::shared_ptr<MetricData> metric_data, bool clockoverlay,
             bool enable_software_trigger, bool enable_resync);
  ~TcamCamera();

  TcamCamera(TcamCamera &) = delete;
  TcamCamera(TcamCamera &&other) : pipeline_{other.pipeline_} {
    other.pipeline_ = nullptr;
  };

  TcamCamera &operator=(const TcamCamera &) = delete;
  TcamCamera &operator=(TcamCamera &&other) {
    gst_object_unref(pipeline_);
    pipeline_ = other.pipeline_;
    other.pipeline_ = nullptr;
    return *this;
  }

  std::string getSerial() { return serial_; }

  /*
   * Get a list of all video formats supported by the device
   */
  std::vector<VideoFormatCaps> get_format_list();
  /*
   * Get a list of all properties supported by the device
   */
  std::vector<std::shared_ptr<Property>> get_camera_property_list();
  /*
   * Get a single camera property
   */
  std::shared_ptr<Property> get_property(std::string name);

  template <typename T>
  std::shared_ptr<T> get_property(std::string name) {
    return std::dynamic_pointer_cast<T>(get_property(name));
  }

  /*
   * Set the video format for capturing
   */
  void set_capture_format(FrameSize size, FrameRate framerate, std::string skipping);
  /*
   * Start capturing video data
   */
  bool start(unsigned long long trigger_start);
  /*
   * Stop capturing video data
   */
  bool stop();
  /*
   * Connect a video display sink element to the capture pipeline
   */
  void enable_video_display(unsigned long win_id);
  /*
   * Disconnect the video display sink element from the capture pipeline
   */
  void disable_video_display();

  void enable_video_udp_stream(std::string host, int port);
  void disable_video_udp_stream();

  /*
   * Connect a record bin to the capture pipeline
   */
  void enable_video_record();
  /*
   * Disconnect the video record bin from the capture pipeline
   */
  void disable_video_record();

  void enable_loop_record();
  void disable_loop_record();

  void enable_video_snapshot();
  void disable_video_snapshot();
  std::shared_ptr<Snapshot> take_snapshot();

  void start_recording(std::string filename, bool enable_image_buf,
                       int imgbuf_width, int imgbuf_height, std::string mode);

  void stop_recording();

  void startLoopRecording();

  void stopLoopRecording();

  /*
   * direction 0 -> identity
   * direction 1 -> 90r
   * direction 3 -> 90l
   */
  void set_direction(int direction);
  int get_direction();

  void setProperty(std::string name, std::string value);

  void resync(unsigned long long t, unsigned long long dt);

 private:
  std::shared_ptr<ImageBuffer> image_queue;

  Module::Monitor *module_monitor = nullptr;
  Module::Record *module_record = nullptr;
  Module::Snapshot *module_snapshot = nullptr;
  Module::Display *module_display = nullptr;
  Module::LoopRecord *module_loop_record = nullptr;
  Module::Udp *module_udp = nullptr;

  std::string serial_ = "";
  GstElement *pipeline_ = nullptr;
  GstElement *tcambin_ = nullptr;
  GstElement *capturecapsfilter_ = nullptr;
  GstElement *tee_ = nullptr;
  GstElement *capturesink_ = nullptr;

  std::vector<VideoFormatCaps> videocaps_;

  void ensure_ready_state();
  void create_fake_pipeline();
  void create_pipeline(bool is_clockoverlay);
  std::vector<VideoFormatCaps> initialize_format_list();

  void static fps_measurements_callback (GstElement * fpsdisplaysink,
                           gdouble fps,
                           gdouble droprate,
                           gdouble avgfps,
                           gpointer udata);

  int direction = 0;

  bool is_fake = false;

  long long clock_offset_ns;

  void initClockOffset();

  bool software_trigger = false;
  std::thread trigger_thread;
  std::chrono::system_clock::time_point trigger_tp;
  std::atomic_bool thread_running = false;
  int trigger_nsec = 0;
  void trigger_func();

  bool enable_resync = false;
  std::thread resync_thread;
  std::atomic_bool is_resync_daemon_running = true;
  guint64 resync_offset = 0; // nsec

  std::atomic_ulong resync_next_timestamp;
  std::atomic_ulong resync_next_dt;
  std::condition_variable resync_cond;
  std::mutex resync_mutex;

  void resync_daemon();
};

std::vector<CameraInfo> get_device_list();

};  // namespace gsttcam

#endif  //__TCAMCAMERA_H__