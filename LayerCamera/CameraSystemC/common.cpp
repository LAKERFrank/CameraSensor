
#include "common.h"
#include <stdexcept>
#include <iostream>

using namespace gsttcam;

ImageBuffer::ImageBuffer() : _slots(MAX_SIZE) {
    for (size_t i = 0; i < MAX_SIZE; ++i) {
        _free_slots.push(i);
    }
}

std::shared_ptr<Frame> ImageBuffer::pop(bool blocking)
{
    size_t consumer_id;
    {
        std::unique_lock<std::mutex> lock(_mutex);
        if (!_legacy_consumer_id.has_value()) {
            size_t id = _consumer_queues.size();
            _consumer_queues.emplace_back();
            _consumer_names.push_back("legacy");
            _legacy_consumer_id = id;
        }
        consumer_id = _legacy_consumer_id.value();
    }

    auto handle = pop_handle(consumer_id, blocking);
    if (!handle) {
        return nullptr;
    }
    auto frame = get(handle);
    release(handle);
    return frame;
}

std::shared_ptr<ImageBuffer::FrameHandle> ImageBuffer::pop_handle(size_t consumer_id, bool blocking)
{
    std::unique_lock<std::mutex> lock(_mutex);
    if (consumer_id >= _consumer_queues.size()) {
        return nullptr;
    }

    auto &queue = _consumer_queues[consumer_id];
    if (blocking) {
        _cond.wait(lock, [&queue]() { return !queue.empty(); });
    } else if (queue.empty()) {
        return nullptr;
    }

    auto ret = queue.front();
    queue.pop();

    std::string consumer_name =
        consumer_id < _consumer_names.size() ? _consumer_names[consumer_id] : std::string("unknown");
    std::cout << "[ImageBuffer] Consumer '" << consumer_name << "' popped frame " << ret->frame_id
              << " from slot " << ret->slot_idx << std::endl;
    return ret;
}

void ImageBuffer::push(std::shared_ptr<Frame> frame)
{
    std::unique_lock<std::mutex> lock(_mutex);

    const auto consumer_cnt = _consumer_queues.size();
    if (consumer_cnt == 0) {
        return;
    }

    _cond.wait(lock, [this]() { return !_free_slots.empty(); });
    const auto slot_idx = _free_slots.front();
    _free_slots.pop();

    FrameSlot &slot = _slots[slot_idx];
    slot.frame = frame;
    slot.frame_id = frame->index;
    slot.refcnt.store(static_cast<int>(consumer_cnt));

    for (auto &consumer_queue : _consumer_queues) {
        auto handle = std::make_shared<FrameHandle>();
        handle->slot_idx = slot_idx;
        handle->frame_id = slot.frame_id;
        consumer_queue.push(handle);
    }

    std::cout << "[ImageBuffer] Pushed frame " << slot.frame_id << " into slot " << slot_idx << " for consumers: ";
    for (size_t i = 0; i < _consumer_names.size(); ++i) {
        std::cout << _consumer_names[i];
        if (i + 1 < _consumer_names.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    _cond.notify_all();
}

std::shared_ptr<Frame> ImageBuffer::get(std::shared_ptr<FrameHandle> handle)
{
    if (!handle) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (handle->slot_idx >= _slots.size()) {
        return nullptr;
    }
    return _slots[handle->slot_idx].frame;
}

void ImageBuffer::release(std::shared_ptr<FrameHandle> handle)
{
    if (!handle) {
        return;
    }

    std::unique_lock<std::mutex> lock(_mutex);
    if (handle->slot_idx >= _slots.size()) {
        return;
    }

    FrameSlot &slot = _slots[handle->slot_idx];
    int ref = --slot.refcnt;
    if (ref == 0) {
        slot.frame.reset();
        _free_slots.push(handle->slot_idx);
        _cond.notify_all();
    }
}

size_t ImageBuffer::register_consumer(const std::string& name)
{
    std::lock_guard<std::mutex> lock(_mutex);
    size_t id = _consumer_queues.size();
    _consumer_queues.emplace_back();
    _consumer_names.push_back(name);
    std::cout << "[ImageBuffer] Registered consumer '" << name << "' with id " << id << std::endl;
    return id;
}

void ImageBuffer::clear()
{
    std::lock_guard<std::mutex> lock(_mutex);

    _consumer_queues.clear();
    _consumer_names.clear();
    std::queue<size_t> empty_slots;
    std::swap(_free_slots, empty_slots);
    for (size_t i = 0; i < MAX_SIZE; ++i) {
        _free_slots.push(i);
    }

    for (auto &slot : _slots) {
        slot.frame.reset();
        slot.refcnt.store(0);
        slot.frame_id = 0;
    }

    _legacy_consumer_id.reset();
}

timespec get_clock_time(clockid_t clock_type) {
    timespec ts;
    if (clock_gettime(clock_type, &ts) != 0) {
        throw std::runtime_error("Failed to get clock time");
    }
    return ts;
}

long long timespec_to_ns(timespec ts) {
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
