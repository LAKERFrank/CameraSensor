#ifndef __MODULE_UDP_H__
#define __MODULE_UDP_H__

#include <iostream>
#include <queue>
#include <memory>
#include <condition_variable>
#include <gst/gst.h>

#define LOG_UDP_MODULE std::cout << "[UDP Module] "

namespace Module
{

class Udp
{
public:
    Udp(GstElement *main_pipeline, GstElement *main_tee, std::string host, int port, int direction);
    ~Udp();

private:
    void createPipeline();
    void start(std::string host, int port, int direction);
    void stop();

    GstElement *_main_bin = nullptr;
    GstElement *_main_tee = nullptr;

    GstElement *_udp_bin = nullptr;
    GstElement *_udp_queue = nullptr;
    GstElement *_udp_flip = nullptr;
    GstElement *_udp_sink = nullptr;

    int _direction = 0;
};

};

#endif
