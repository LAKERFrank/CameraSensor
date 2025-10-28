#ifndef __MODULE_DISPLAY_H__
#define __MODULE_DISPLAY_H__

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

#define LOG_DISPLAY_MODULE std::cout << "[Display Module] "

namespace Module {

class Display {
public:
    Display(GstElement *main_pipeline, GstElement *main_tee, unsigned long win_id, int direction);
    ~Display();
private:
    void createPipeline(GstElement *displaysink);

    GstElement *_main_bin = nullptr;
    GstElement *_main_tee = nullptr;

    GstElement *_display_bin = nullptr;
    GstElement *_display_queue = nullptr;

    int _direction = 0;
};

};

#endif
