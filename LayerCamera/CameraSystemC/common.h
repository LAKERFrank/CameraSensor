#ifndef __COMMON_H__
#define __COMMON_H__

#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <opencv4/opencv2/opencv.hpp>

timespec get_clock_time(clockid_t clock_type);
long long timespec_to_ns(timespec ts);

namespace gsttcam {

class Frame : public std::enable_shared_from_this<Frame> {
public:
    Frame() {};
    ~Frame() {};

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getIndex() const { return index; }
    double getTimestamp() const { return timestamp; }
    double getMonotonicTimestamp() const { return monotonic_timestamp; }
    void setIsEOS(bool is_eos) { this->is_eos = is_eos; }
    bool getIsEOS() const { return is_eos; }

    bool is_eos = false;
    int index = 0;
    int width = 0;
    int height = 0;
    std::shared_ptr<u_int8_t> memory = nullptr;
    unsigned long memory_size = 0;
    double timestamp;
    double monotonic_timestamp;
};

class ImageBuffer : public std::enable_shared_from_this<ImageBuffer>
{
public:
    ImageBuffer() {};
    ~ImageBuffer() {};
    std::shared_ptr<Frame> pop(bool blocking = true);
    void push(std::shared_ptr<Frame> frame);
    void clear();

private:
    unsigned int MAX_SIZE = 600; // 5s under 120fps
    std::queue<std::shared_ptr<Frame>> _image_queue;
    std::mutex _mutex;

    // Condition variable for signaling
    std::condition_variable _cond;
};

class Snapshot : public std::enable_shared_from_this<Snapshot>{
public:
  Snapshot() {};
  ~Snapshot() {};
  unsigned int width;
  unsigned int height;
  unsigned int memory_size;
  std::shared_ptr<u_int8_t> memory = nullptr;

  unsigned int getWidth() const { return width; }
  unsigned int getHeight() const { return height; }
  unsigned int getMemorySize() const { return memory_size; }
  std::shared_ptr<u_int8_t> getMemory() const { return memory; }

  std::shared_ptr<cv::Mat> toCvMat() const {
    return std::make_shared<cv::Mat>(height, width, CV_8UC3, memory.get());
  }
};

class MetricData : public std::enable_shared_from_this<MetricData> {
public:
  MetricData() {
    frames_dropped = 0;
    frames_rendered = 0;
    fps = 0.0f;
    avg_fps = 0.0f;
  };
  unsigned int frames_dropped;
  unsigned int frames_rendered;
  double fps;
  double avg_fps;

  void set_kf(double timestamp, double dt) {
    std::lock_guard<std::mutex> lock(kf_mutex);
    kf_timestamp = timestamp;
    kf_dt = dt;
  }

  void get_kf(double& timestamp, double& dt) {
    std::lock_guard<std::mutex> lock(kf_mutex);
    timestamp = kf_timestamp;
    dt = kf_dt;
  }

private:
  double kf_timestamp;
  double kf_dt;
  std::mutex kf_mutex;
};

};

#endif
