#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "common.h"

namespace camerasensor
{

struct Timestamp
{
    std::chrono::steady_clock::time_point tp{};
};

struct SharedFrame
{
    uint64_t frame_id = 0;
    int camera_id = 0;
    Timestamp ts{};
    std::shared_ptr<gsttcam::Frame> frame_payload;

    std::atomic<int> ref_count{0};
};

struct FrameHandle
{
    SharedFrame* frame;

    FrameHandle() : frame(nullptr) {}

    explicit FrameHandle(SharedFrame* f) : frame(f)
    {
        if (frame)
        {
            frame->ref_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    FrameHandle(const FrameHandle& other) : frame(other.frame)
    {
        if (frame)
        {
            frame->ref_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    FrameHandle(FrameHandle&& other) noexcept : frame(other.frame) { other.frame = nullptr; }

    FrameHandle& operator=(const FrameHandle& other)
    {
        if (this == &other)
            return *this;
        release();
        frame = other.frame;
        if (frame)
        {
            frame->ref_count.fetch_add(1, std::memory_order_relaxed);
        }
        return *this;
    }

    FrameHandle& operator=(FrameHandle&& other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        frame = other.frame;
        other.frame = nullptr;
        return *this;
    }

    ~FrameHandle() { release(); }

private:
    void release()
    {
        if (frame)
        {
            if (frame->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                // Slot will be reused by the SharedFramePool; do not delete here.
            }
            frame = nullptr;
        }
    }
};

// Simple blocking queue used by the two model branches.
template <typename T>
class ThreadSafeQueue
{
public:
    void push(const T& value)
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(value);
        }
        cv_.notify_one();
    }

    bool pop(T& out)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return !queue_.empty() || stopped_; });
        if (queue_.empty())
        {
            return false;
        }
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stopped_{false};
};

class SharedFramePool
{
public:
    explicit SharedFramePool(size_t capacity);

    SharedFrame* acquire_writable_slot();
    void commit_written_slot(SharedFrame* slot);

private:
    std::vector<SharedFrame> buffer_;
    size_t capacity_;
    std::atomic<size_t> write_index_;
};

struct RawImage
{
    std::shared_ptr<gsttcam::Frame> frame_payload;
};

class CameraSensor
{
public:
    using GrabFn = std::function<RawImage()>;
    using TimestampFn = std::function<Timestamp()>;
    using ConvertFn = std::function<std::shared_ptr<gsttcam::Frame>(const RawImage&)>;

    CameraSensor(int camera_id,
                 size_t frame_pool_capacity,
                 ThreadSafeQueue<FrameHandle>& tracknet_input_queue,
                 ThreadSafeQueue<FrameHandle>& pose_raw_queue,
                 GrabFn grab_fn,
                 TimestampFn ts_fn,
                 ConvertFn convert_fn);

    void capture_loop();
    void stop();

private:
    RawImage grab_from_camera();
    Timestamp now_timestamp();
    std::shared_ptr<gsttcam::Frame> convert_raw_to_frame(const RawImage& raw);

    std::atomic<bool> running_{true};
    int camera_id_{0};
    SharedFramePool frame_pool_;
    ThreadSafeQueue<FrameHandle>& tracknet_input_queue_;
    ThreadSafeQueue<FrameHandle>& pose_raw_queue_;

    GrabFn grab_fn_;
    TimestampFn ts_fn_;
    ConvertFn convert_fn_;
};

class TrackNetWorker
{
public:
    using TrackNetFn = std::function<void(const std::array<std::shared_ptr<gsttcam::Frame>, 10>& batch)>;

    TrackNetWorker(ThreadSafeQueue<FrameHandle>& input_queue, TrackNetFn tracknet_fn);

    void run();
    void stop();

private:
    std::atomic<bool> running_{true};
    ThreadSafeQueue<FrameHandle>& input_queue_;
    TrackNetFn tracknet_fn_;
};

class PoseFrameSelector
{
public:
    PoseFrameSelector(ThreadSafeQueue<FrameHandle>& raw_queue,
                      ThreadSafeQueue<FrameHandle>& pose_queue,
                      int camera_fps,
                      int pose_fps);

    void run();
    void stop();

private:
    std::atomic<bool> running_{true};
    ThreadSafeQueue<FrameHandle>& raw_queue_;
    ThreadSafeQueue<FrameHandle>& pose_queue_;
    int camera_fps_;
    int pose_fps_;
    int frame_step_;
    uint64_t counter_{0};
};

class PoseWorker
{
public:
    using PoseFn = std::function<void(const std::shared_ptr<gsttcam::Frame>&)>;

    PoseWorker(ThreadSafeQueue<FrameHandle>& input_queue, PoseFn pose_fn);

    void run();
    void stop();

private:
    std::atomic<bool> running_{true};
    ThreadSafeQueue<FrameHandle>& input_queue_;
    PoseFn pose_fn_;
};

class CameraPipeline
{
public:
    CameraPipeline(int camera_id,
                   size_t frame_pool_capacity,
                   int camera_fps,
                   int pose_fps,
                   CameraSensor::GrabFn grab_fn,
                   CameraSensor::TimestampFn ts_fn,
                   CameraSensor::ConvertFn convert_fn,
                   TrackNetWorker::TrackNetFn tracknet_fn,
                   PoseWorker::PoseFn pose_fn);

    ~CameraPipeline();

    void start();
    void stop();

private:
    void join_threads();

    ThreadSafeQueue<FrameHandle> tracknet_input_queue_;
    ThreadSafeQueue<FrameHandle> pose_raw_queue_;
    ThreadSafeQueue<FrameHandle> pose_input_queue_;

    CameraSensor camera_sensor_;
    TrackNetWorker tracknet_worker_;
    PoseFrameSelector pose_selector_;
    PoseWorker pose_worker_;

    std::thread capture_thread_;
    std::thread tracknet_thread_;
    std::thread pose_selector_thread_;
    std::thread pose_thread_;
};

} // namespace camerasensor

