#ifndef __MODULE_MONITOR_H__
#define __MODULE_MONITOR_H__

#include <iostream>
#include <queue>
#include <memory>
#include <condition_variable>
#include <gst/gst.h>

#include "gstmetatcamstatistics.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include "../common.h"
#include "../kalmanfilter.h"

#define LOG_MONITOR_MODULE std::cout << "[Monitor Module] "

namespace Module {

class Monitor {
public:
    Monitor(GstElement *main_pipeline, GstElement *main_tee, std::shared_ptr<gsttcam::MetricData> metric_data);
    ~Monitor();

    void setClockOffset(long long v);

    guint64 record_t = 0;
    guint64 first_t = 0;
private:

    void createPipeline();

    void static fpsCallback(GstElement * fpsdisplaysink,
                           gdouble fps,
                           gdouble droprate,
                           gdouble avgfps,
                           gpointer udata);

    static GstFlowReturn imageCallback(GstElement *sink, gpointer user_data);

    GstElement *_mainBin = nullptr;
    GstElement *_mainTee = nullptr;

    GstElement *_monitorBin = nullptr;
    GstElement *_monitorQueue = nullptr;

    SimpleKalmanFilter _kf;

    std::shared_ptr<gsttcam::MetricData> _metricData;

    // record offset between boottime & realtime clock
    long long clockOffset;
};

};

#endif // __MODULE_MONITOR_H__
