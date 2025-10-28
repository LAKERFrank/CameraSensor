#include "display.h"

namespace Module
{

Display::Display(GstElement *main_pipeline, GstElement *main_tee, unsigned long win_id, int direction)
{
    this->_main_bin = main_pipeline;
    this->_main_tee = main_tee;
    this->_direction = direction;

    GstElement *fpsdisplaysink =
        gst_element_factory_make("fpsdisplaysink", NULL);
    GstElement *imagesink = gst_element_factory_make("ximagesink", NULL);

    g_object_set(GST_BIN(fpsdisplaysink), "video-sink", imagesink, "sync",
                 false, "text-overlay", true, NULL);

    // WId xwinid = videoWidgets[i]->winId();
    // Assing the window handle
    // Pass the display sink to the TcamCamera object

    LOG_DISPLAY_MODULE << "winId -> " << win_id << std::endl;

    gst_video_overlay_set_window_handle(GST_VIDEO_OVERLAY(imagesink), win_id);

    this->createPipeline(fpsdisplaysink);

    gst_element_set_state(_display_bin, GST_STATE_PLAYING);

    gst_bin_add(GST_BIN(_main_bin), _display_bin);
    gst_element_link(_main_tee, _display_queue);

    LOG_DISPLAY_MODULE << "Start display" << std::endl;

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_main_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[display_module]start(main)");
}

Display::~Display()
{
    gst_element_unlink(_main_tee, _display_queue);
    gst_bin_remove(GST_BIN(_main_bin), _display_bin);
    gst_element_set_state(_display_bin, GST_STATE_NULL);
}

void Display::createPipeline(GstElement *displaysink)
{
    if (_display_bin) {
        return;
    }
    _display_bin = gst_element_factory_make("bin", nullptr);
    _display_queue = gst_element_factory_make("queue", nullptr);
    g_object_set(G_OBJECT(_display_queue), "leaky", 1, nullptr);

    GstElement *videorate = gst_element_factory_make("videorate", nullptr);
    GstElement *capsfilter = gst_element_factory_make("capsfilter", nullptr);

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                        "framerate", GST_TYPE_FRACTION, 15, 1, nullptr);

    g_object_set(G_OBJECT(capsfilter), "caps", caps, nullptr);
    gst_caps_unref(caps);

    GstElement *videoflip = gst_element_factory_make("videoflip", nullptr);

    g_object_set(G_OBJECT(videoflip), "video-direction", _direction, nullptr);

    GstElement *videoscale = gst_element_factory_make("videoscale", "Display_videoscale");
    GstElement *convert = gst_element_factory_make("videoconvert", nullptr);
    gst_bin_add_many(GST_BIN(_display_bin), _display_queue, videorate, videoscale, capsfilter,
                     videoflip, convert, displaysink, nullptr);
    if (!gst_element_link_many(_display_queue, videorate, videoscale, capsfilter, videoflip,
                               convert, displaysink, nullptr))
        throw std::runtime_error("Could not link elements");

    gst_element_set_state(_display_bin, GST_STATE_READY);
}

};
