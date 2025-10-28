#include "snapshot.h"

namespace Module {

Snapshot::Snapshot(GstElement *main_pipeline, GstElement *main_tee, int direction)
{
    this->_main_bin = main_pipeline;
    this->_main_tee = main_tee;
    this->_direction = direction;

    this->createPipeline();
}

Snapshot::~Snapshot()
{
    gst_element_unlink(_main_tee, _snapshot_queue);
    gst_bin_remove(GST_BIN(_main_bin), _snapshot_bin);
    gst_element_set_state(_snapshot_bin, GST_STATE_NULL);
}

void Snapshot::createPipeline()
{
    if (_snapshot_bin)
    {
        return;
    }

    _snapshot_bin = gst_element_factory_make("bin", nullptr);
    _snapshot_queue = gst_element_factory_make("queue", nullptr);
    g_object_set(G_OBJECT(_snapshot_queue), "leaky", 1, nullptr);

    GstElement *videorate = gst_element_factory_make("videorate", nullptr);
    GstElement *capsfilter = gst_element_factory_make("capsfilter", nullptr);

    GstCaps *caps = gst_caps_new_simple("video/x-raw", "framerate",
                                        GST_TYPE_FRACTION, 5, 1, nullptr);

    g_object_set(G_OBJECT(capsfilter), "caps", caps, nullptr);
    gst_caps_unref(caps);

    GstElement *videoflip = gst_element_factory_make("videoflip", nullptr);

    g_object_set(G_OBJECT(videoflip), "video-direction", _direction, nullptr);

    GstElement *convert = gst_element_factory_make("videoconvert", nullptr);
    GstElement *capsfilter2 = gst_element_factory_make("capsfilter", nullptr);
    GstElement *appsink = gst_element_factory_make("appsink", nullptr);
    g_object_set(G_OBJECT(appsink), "sync", false, nullptr);

    // make format=BGR
    GstCaps *caps2 = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR", nullptr);
    g_object_set(G_OBJECT(capsfilter2), "caps", caps2, nullptr);
    gst_caps_unref(caps2);

    g_object_set(G_OBJECT(appsink), "drop", true, nullptr);
    g_object_set(G_OBJECT(appsink), "max-buffers", 1, nullptr);

    g_object_set(G_OBJECT(appsink), "emit-signals", TRUE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(snapshot_callback), this);

    gst_bin_add(GST_BIN(_main_bin), _snapshot_bin);
    gst_bin_add_many(GST_BIN(_snapshot_bin), _snapshot_queue, videorate, capsfilter, videoflip,
                     convert, capsfilter2, appsink, nullptr);
    if (!gst_element_link_many(_main_tee, _snapshot_queue, videorate, capsfilter, videoflip,
                               convert, capsfilter2, appsink, nullptr))
        throw std::runtime_error("Could not link elements");
}

std::shared_ptr<gsttcam::Snapshot> Snapshot::take()
{
  std::unique_lock<std::mutex> lock(_take_mutex);

  // clear previous image
  _snapshot = nullptr;

  _take_cond.wait(lock, [&]() { return _snapshot != nullptr; });

  return _snapshot;
}

GstFlowReturn Snapshot::snapshot_callback(GstElement *sink, gpointer user_data)
{
    Snapshot *self = static_cast<Snapshot *>(user_data);

    // lock below critical section
    std::lock_guard<std::mutex> lock(self->_take_mutex);

    if (self->_snapshot != nullptr)
    {
        // LOGMYSYSTEM << "snapshot callback skip\n";
        return GST_FLOW_OK;
    }

    GstSample *sample = NULL;
    /* Retrieve the buffer */
    g_signal_emit_by_name(sink, "pull-sample", &sample, NULL);

    if (sample)
    {
        GstBuffer *buffer = gst_sample_get_buffer(sample);

        GstCaps *caps = gst_sample_get_caps(sample);

        GstStructure *caps_struct = gst_caps_get_structure(caps, 0);
        if (!caps_struct)
        {
            g_warning("caps have NULL structure");
        }

        GstMapInfo info;
        if (!gst_buffer_map(buffer, &info, GST_MAP_READ))
        {
            return GST_FLOW_ERROR;
        }

        GstVideoInfo* video_info = gst_video_info_new();
        if (!gst_video_info_from_caps(video_info, gst_sample_get_caps(sample)))
        {
            // Could not parse video info (should not happen)
            g_warning("Failed to parse video info");
            return GST_FLOW_ERROR;
        }

        self->_snapshot = std::make_shared<gsttcam::Snapshot>();
        self->_snapshot->width = video_info->width;
        self->_snapshot->height = video_info->height;
        self->_snapshot->memory_size = info.size;
        self->_snapshot->memory = std::shared_ptr<u_int8_t>((u_int8_t *)malloc(info.size), free);
        memcpy(self->_snapshot->memory.get(), info.data, info.size);

        // delete our reference so that gstreamer can handle the sample
        gst_video_info_free(video_info);
        gst_buffer_unmap(buffer, &info);
        gst_sample_unref(sample);

        self->_take_cond.notify_all();
    }
    return GST_FLOW_OK;
}
};
