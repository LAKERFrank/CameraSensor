#include "monitor.h"

namespace Module
{

Monitor::Monitor(GstElement *main_pipeline, GstElement *main_tee, std::shared_ptr<gsttcam::MetricData> metric_data)
{   
    this->_mainBin = main_pipeline;
    this->_mainTee = main_tee;
    this->_metricData = metric_data;

    this->createPipeline();

    gst_element_set_state(_monitorBin, GST_STATE_PLAYING);

    gst_bin_add(GST_BIN(_mainBin), _monitorBin);
    gst_element_link(_mainTee, _monitorQueue);

    LOG_MONITOR_MODULE << "Start monitor" << std::endl;
};

Monitor::~Monitor()
{
    gst_element_unlink(_mainTee, _monitorQueue);
    gst_bin_remove(GST_BIN(_mainBin), _monitorBin);
    gst_element_set_state(_monitorBin, GST_STATE_NULL);
};

void Monitor::createPipeline()
{
    if (_monitorBin) {
        return;
    }
    _monitorBin = gst_element_factory_make("bin", nullptr);
    _monitorQueue = gst_element_factory_make("queue", nullptr);

    // Link FPS measurement sink
    GstElement *fpssink = gst_element_factory_make("fpsdisplaysink", "fps_measurements_sink");
    GstElement *sink = gst_element_factory_make("appsink", "imgbuf_sink");
    g_object_set(G_OBJECT(sink), "emit-signals", TRUE, NULL);
    g_signal_connect(sink, "new-sample", G_CALLBACK(imageCallback), this);
    //GstElement *sink = gst_element_factory_make("fakesink", nullptr);
    g_object_set(G_OBJECT(fpssink), "sync", false, nullptr);
    g_object_set(G_OBJECT(fpssink), "signal-fps-measurements", true, nullptr);
    g_object_set(G_OBJECT(fpssink), "text-overlay", false, nullptr);
    g_object_set(G_OBJECT(fpssink), "video-sink", sink, nullptr);
    g_signal_connect(fpssink, "fps-measurements", G_CALLBACK(fpsCallback), this);

    assert(_monitorBin && _monitorQueue && fpssink && sink);

    gst_bin_add_many(GST_BIN(_monitorBin), _monitorQueue, fpssink, nullptr);
    if (!gst_element_link_many(_monitorQueue, fpssink, nullptr))
        throw std::runtime_error("Could not link elements");

    gst_element_set_state(_monitorBin, GST_STATE_READY);
};

void Monitor::fpsCallback (GstElement *fpsdisplaysink, gdouble fps, gdouble droprate, gdouble avgfps,
                                            gpointer udata)
{
    Monitor *m = static_cast<Monitor *>(udata);

    guint frames_dropped, frames_rendered;
    g_object_get(G_OBJECT(fpsdisplaysink), "frames-dropped", &frames_dropped, NULL);
    g_object_get(G_OBJECT(fpsdisplaysink), "frames-rendered", &frames_rendered, NULL);

    m->_metricData->fps = fps;
    m->_metricData->avg_fps = avgfps;
    m->_metricData->frames_dropped = frames_dropped;
    m->_metricData->frames_rendered = frames_rendered;
};

GstFlowReturn Monitor::imageCallback(GstElement *sink, gpointer user_data)
{
    Monitor *m = static_cast<Monitor *>(user_data);

    GstSample *sample = NULL;
    // Retrieve the buffer
    g_signal_emit_by_name(sink, "pull-sample", &sample, NULL);

    if (sample)
    {
        GstBuffer *buffer = gst_sample_get_buffer(sample);

        GstMeta *meta = gst_buffer_get_meta(buffer, g_type_from_name("TcamStatisticsMetaApi"));

        if (meta)
        {
            GstStructure *struc = ((TcamStatisticsMeta *)meta)->structure;
            guint64 boottime_nsec;
            gst_structure_get(struc, "capture_time_ns", G_TYPE_UINT64, &boottime_nsec, NULL);
            guint64 realtime_nsec = boottime_nsec + m->clockOffset;
            // TODO:
            if (m->record_t != 0 && realtime_nsec > m->record_t && m->first_t == 0) {
                // record fist frame timestamp after record_t
                m->first_t = realtime_nsec;
            }

            m->_metricData->set_kf(m->_kf.timestamp(), m->_kf.dt());

            double realtime_sec = (double)realtime_nsec / 1'000'000'000;

            m->_kf.step(realtime_sec);
        }
        else
        {
            g_warning("No meta data available");
        }

        // delete our reference so that gstreamer can handle the sample
        gst_sample_unref(sample);
    }
    return GST_FLOW_OK;
};

void Monitor::setClockOffset(long long v) {
    this->clockOffset = v;
}

};
