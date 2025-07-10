#ifndef __MODULE_RECORD_H__
#define __MODULE_RECORD_H__

#include <iostream>
#include <queue>
#include <memory>
#include <condition_variable>
#include <filesystem>
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

#define LOG_RECORD_MODULE std::cout << "[Record Module] "

namespace Module {

enum class RecordMode {
    NONE,
    H264_LOW,
    H264_HIGH,
    LOSSLESS
};

class Record {
public:
    Record(GstElement *main_pipeline, GstElement *main_tee,
        std::shared_ptr<gsttcam::ImageBuffer> imgbuf);
    ~Record() {};

    void setClockOffset(long long v);

    void startRecording(std::string filename, bool is_imgbuf_enabled,
                        int imgbuf_width, int imgbuf_height, std::string mode);
    void stopRecording();

private:
    void createPipeline();
    void createH264Bin();
    void createLosslessBin();
    void reset();

    RecordMode mode = RecordMode::NONE;

    GstElement *_main_bin = nullptr;
    GstElement *_main_tee = nullptr;

    GstElement *_record_bin = nullptr;
    GstElement *_record_queue = nullptr;
    GstElement *_record_tee = nullptr;

    GstElement *_imgbuf_capsfilter = nullptr;

    GstElement *_file_h264_bin = nullptr;
    GstElement *_file_h264_encoder = nullptr;
    GstElement *_file_h264_sink = nullptr;
    GstElement *_file_lossless_bin = nullptr;
    GstElement *_file_lossless_encoder = nullptr;
    GstElement *_file_lossless_sink = nullptr;

    // not modified under reset
    bool _enable_image_buf = false;
    std::shared_ptr<gsttcam::ImageBuffer> _imgbuf;

    bool is_recording = false;

    gint frame_idx = 0;

    std::ofstream meta_csv_;

    static GstFlowReturn imgbuf_callback(GstElement* sink, gpointer user_data);
    static void imgbuf_eos_callback (GstElement *, gpointer user_data);
    static GstPadProbeReturn record_eos_callback(GstPad *, GstPadProbeInfo *info, gpointer user_data);

    bool is_record_eos = false;
    bool is_imgbuf_eos = false;
    std::condition_variable eos_cond;
    std::mutex eos_mutex;

    // record offset between boottime & realtime clock
    long long clock_offset;
};
};

#endif
