#include "shared_frame_pipeline.h"

namespace camerasensor
{

SharedFramePool::SharedFramePool(size_t capacity)
    : buffer_(capacity), capacity_(capacity), write_index_(0)
{
}

SharedFrame* SharedFramePool::acquire_writable_slot()
{
    while (true)
    {
        size_t idx = write_index_.load(std::memory_order_relaxed) % capacity_;
        SharedFrame& slot = buffer_[idx];

        int expected = 0;
        if (slot.ref_count.compare_exchange_strong(expected, 0, std::memory_order_acq_rel,
                                                   std::memory_order_relaxed))
        {
            return &slot;
        }

        write_index_.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    }
}

void SharedFramePool::commit_written_slot(SharedFrame* /*slot*/)
{
    write_index_.fetch_add(1, std::memory_order_relaxed);
}

CameraSensor::CameraSensor(int camera_id,
                           size_t frame_pool_capacity,
                           ThreadSafeQueue<FrameHandle>& tracknet_input_queue,
                           ThreadSafeQueue<FrameHandle>& pose_raw_queue,
                           GrabFn grab_fn,
                           TimestampFn ts_fn,
                           ConvertFn convert_fn)
    : camera_id_(camera_id),
      frame_pool_(frame_pool_capacity),
      tracknet_input_queue_(tracknet_input_queue),
      pose_raw_queue_(pose_raw_queue),
      grab_fn_(std::move(grab_fn)),
      ts_fn_(std::move(ts_fn)),
      convert_fn_(std::move(convert_fn))
{
}

void CameraSensor::capture_loop()
{
    uint64_t global_frame_id = 0;

    while (running_)
    {
        RawImage raw = grab_from_camera();
        SharedFrame* slot = frame_pool_.acquire_writable_slot();

        slot->frame_id = global_frame_id++;
        slot->camera_id = camera_id_;
        slot->ts = now_timestamp();
        // Reuse the already allocated camera frame buffer; do not duplicate image memory here.
        slot->frame_payload = convert_raw_to_frame(raw);

        frame_pool_.commit_written_slot(slot);

        FrameHandle h_track(slot);
        FrameHandle h_pose(slot);

        tracknet_input_queue_.push(h_track);
        pose_raw_queue_.push(h_pose);
    }

    tracknet_input_queue_.stop();
    pose_raw_queue_.stop();
}

void CameraSensor::stop()
{
    running_ = false;
}

RawImage CameraSensor::grab_from_camera()
{
    if (grab_fn_)
    {
        return grab_fn_();
    }
    return {};
}

Timestamp CameraSensor::now_timestamp()
{
    if (ts_fn_)
    {
        return ts_fn_();
    }
    return {std::chrono::steady_clock::now()};
}

std::shared_ptr<gsttcam::Frame> CameraSensor::convert_raw_to_frame(const RawImage& raw)
{
    if (convert_fn_)
    {
        return convert_fn_(raw);
    }
    return raw.frame_payload;
}

TrackNetWorker::TrackNetWorker(ThreadSafeQueue<FrameHandle>& input_queue, TrackNetFn tracknet_fn)
    : input_queue_(input_queue), tracknet_fn_(std::move(tracknet_fn))
{
}

void TrackNetWorker::run()
{
    const int kWindowSize = 10;
    const int kSlideStep = 1;
    std::deque<FrameHandle> window;

    while (running_)
    {
        FrameHandle h;
        if (!input_queue_.pop(h))
        {
            break;
        }

        window.push_back(std::move(h));

        if (static_cast<int>(window.size()) < kWindowSize)
        {
            continue;
        }

        std::array<std::shared_ptr<gsttcam::Frame>, kWindowSize> batch;
        int i = 0;
        for (auto& fh : window)
        {
            batch[i++] = fh.frame->frame_payload;
        }

        if (tracknet_fn_)
        {
            tracknet_fn_(batch);
        }

        for (int step = 0; step < kSlideStep && !window.empty(); ++step)
        {
            window.pop_front();
        }
    }
}

void TrackNetWorker::stop()
{
    running_ = false;
    input_queue_.stop();
}

PoseFrameSelector::PoseFrameSelector(ThreadSafeQueue<FrameHandle>& raw_queue,
                                     ThreadSafeQueue<FrameHandle>& pose_queue,
                                     int camera_fps,
                                     int pose_fps)
    : raw_queue_(raw_queue),
      pose_queue_(pose_queue),
      camera_fps_(camera_fps),
      pose_fps_(pose_fps),
      frame_step_(std::max(1, camera_fps_ / pose_fps_))
{
}

void PoseFrameSelector::run()
{
    while (running_)
    {
        FrameHandle h;
        if (!raw_queue_.pop(h))
        {
            break;
        }

        if ((counter_++ % frame_step_) == 0)
        {
            pose_queue_.push(h);
        }
    }
}

void PoseFrameSelector::stop()
{
    running_ = false;
    raw_queue_.stop();
}

PoseWorker::PoseWorker(ThreadSafeQueue<FrameHandle>& input_queue, PoseFn pose_fn)
    : input_queue_(input_queue), pose_fn_(std::move(pose_fn))
{
}

void PoseWorker::run()
{
    while (running_)
    {
        FrameHandle h;
        if (!input_queue_.pop(h))
        {
            break;
        }

        if (pose_fn_)
        {
            pose_fn_(h.frame->frame_payload);
        }
    }
}

void PoseWorker::stop()
{
    running_ = false;
    input_queue_.stop();
}

CameraPipeline::CameraPipeline(int camera_id,
                               size_t frame_pool_capacity,
                               int camera_fps,
                               int pose_fps,
                               CameraSensor::GrabFn grab_fn,
                               CameraSensor::TimestampFn ts_fn,
                               CameraSensor::ConvertFn convert_fn,
                               TrackNetWorker::TrackNetFn tracknet_fn,
                               PoseWorker::PoseFn pose_fn)
    : camera_sensor_(camera_id,
                     frame_pool_capacity,
                     tracknet_input_queue_,
                     pose_raw_queue_,
                     std::move(grab_fn),
                     std::move(ts_fn),
                     std::move(convert_fn)),
      tracknet_worker_(tracknet_input_queue_, std::move(tracknet_fn)),
      pose_selector_(pose_raw_queue_, pose_input_queue_, camera_fps, pose_fps),
      pose_worker_(pose_input_queue_, std::move(pose_fn))
{
}

CameraPipeline::~CameraPipeline()
{
    stop();
    join_threads();
}

void CameraPipeline::start()
{
    capture_thread_ = std::thread([this]() { camera_sensor_.capture_loop(); });
    tracknet_thread_ = std::thread([this]() { tracknet_worker_.run(); });
    pose_selector_thread_ = std::thread([this]() { pose_selector_.run(); });
    pose_thread_ = std::thread([this]() { pose_worker_.run(); });
}

void CameraPipeline::stop()
{
    camera_sensor_.stop();
    tracknet_worker_.stop();
    pose_selector_.stop();
    pose_worker_.stop();
}

void CameraPipeline::join_threads()
{
    if (capture_thread_.joinable())
    {
        capture_thread_.join();
    }
    if (tracknet_thread_.joinable())
    {
        tracknet_thread_.join();
    }
    if (pose_selector_thread_.joinable())
    {
        pose_selector_thread_.join();
    }
    if (pose_thread_.joinable())
    {
        pose_thread_.join();
    }
}

} // namespace camerasensor

