#ifndef __MODULE_SNAPSHOT_H__
#define __MODULE_SNAPSHOT_H__

#include <iostream>
#include <queue>
#include <memory>
#include <condition_variable>
#include <gst/gst.h>

#include "gstmetatcamstatistics.h"

#include <gst/gst.h>
#include <gst/video/video.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <iomanip>


#include "../common.h"

#define LOG_SNAPSHOT_MODULE std::cout << "[Snapshot Module] "

namespace Module {

class Snapshot {
public:
    Snapshot(GstElement *main_pipeline, GstElement *main_tee, int direction);
    ~Snapshot();

    std::shared_ptr<gsttcam::Snapshot> take();
private:
    void createPipeline();

    static GstFlowReturn snapshot_callback(GstElement* sink, gpointer user_data);

    GstElement *_main_bin = nullptr;
    GstElement *_main_tee = nullptr;

    GstElement *_snapshot_bin = nullptr;
    GstElement *_snapshot_queue = nullptr;

    std::condition_variable _take_cond;
    std::mutex _take_mutex;

    std::shared_ptr<gsttcam::Snapshot> _snapshot = nullptr;

    int _direction = 0;
};

};

#endif