
#include "common.h"

using namespace gsttcam;

std::shared_ptr<Frame> ImageBuffer::pop(bool blocking)
{
    std::unique_lock<std::mutex> lock(_mutex);

    // wait until queue is not empty
    if (blocking) {
        _cond.wait(lock, [this](){ return !_image_queue.empty(); });
    }
    else if (_image_queue.empty()) {
        return nullptr;
    }

    std::shared_ptr<Frame> ret = _image_queue.front();

    _image_queue.pop();
    return ret;
}

void ImageBuffer::push(std::shared_ptr<Frame> frame)
{
    std::unique_lock<std::mutex> lock(_mutex);

    _image_queue.push(frame);

    if (_image_queue.size() > MAX_SIZE) {
        std::cerr << "ImageQueue dropped 1 frame." << std::endl;
        _image_queue.pop();
    }

    _cond.notify_one();
}
void ImageBuffer::clear()
{
    std::queue<std::shared_ptr<Frame>> empty_queue;

    _image_queue.swap(empty_queue);
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
