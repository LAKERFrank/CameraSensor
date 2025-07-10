#include "udp.h"

namespace Module
{
Udp::Udp(GstElement *main_pipeline, GstElement *main_tee, std::string host, int port, int direction)
{
    this->_main_bin = main_pipeline;
    this->_main_tee = main_tee;
    this->createPipeline();
    this->start(host, port, direction);
}
Udp::~Udp() {
    this->stop();
};

void Udp::start(std::string host, int port, int direction)
{
    LOG_UDP_MODULE << host << ":" << port << std::endl;
    g_object_set(G_OBJECT(_udp_sink), "host", host.c_str(), nullptr);
    g_object_set(G_OBJECT(_udp_sink), "port", port, nullptr);

    g_object_set(G_OBJECT(_udp_flip), "video-direction", direction, nullptr);

    gst_bin_add(GST_BIN(_main_bin), _udp_bin);

    gst_element_set_state(_udp_bin, GST_STATE_PLAYING);

    if (!gst_element_link(_main_tee, _udp_queue)) {
        std::cerr << "ERROR linking elements" << std::endl;
    }
}

void Udp::stop()
{
    gst_element_unlink(_main_tee, _udp_queue);
    gst_bin_remove(GST_BIN(_main_bin), _udp_bin);

    gst_element_set_state(_udp_bin, GST_STATE_NULL);
}

void Udp::createPipeline()
{
    if (_udp_bin)
        return;
    _udp_bin = gst_element_factory_make("bin", nullptr);
    _udp_queue = gst_element_factory_make("queue", nullptr);
    g_object_set(G_OBJECT(_udp_queue), "leaky", 1, nullptr);

    GstElement *videorate = gst_element_factory_make("videorate", nullptr);
    g_object_set(G_OBJECT(videorate), "drop-only", true, nullptr);
    GstElement *capsfilter = gst_element_factory_make("capsfilter", nullptr);

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                      "width", G_TYPE_INT, 320,
                                      "height", G_TYPE_INT, 240,
                                      "framerate", GST_TYPE_FRACTION, 60, 1, nullptr);

    g_object_set(G_OBJECT(capsfilter), "caps", caps, nullptr);
    gst_caps_unref(caps);

    _udp_flip = gst_element_factory_make("videoflip", nullptr);

    GstElement *scale = gst_element_factory_make("videoscale", nullptr);
    GstElement *convert = gst_element_factory_make("videoconvert", nullptr);
    GstElement *encoder = gst_element_factory_make("nvh264enc", nullptr);
    g_object_set(G_OBJECT(encoder), "bitrate", 2000, nullptr);
    g_object_set(G_OBJECT(encoder), "tune", 0x00000004, nullptr); // (0x00000004): zerolatency
    g_object_set(G_OBJECT(encoder), "speed-preset", 1, nullptr); // (1): ultrafast
    GstElement *parser = gst_element_factory_make("h264parse", nullptr);
    GstElement *pay = gst_element_factory_make("rtph264pay", nullptr);
    g_object_set(G_OBJECT(pay), "pt", 96, nullptr);
    g_object_set(G_OBJECT(pay), "config-interval", 1, nullptr);
    _udp_sink = gst_element_factory_make("udpsink", nullptr);

    g_object_set(G_OBJECT(_udp_sink), "sync", false, nullptr);

    gst_bin_add_many(GST_BIN(_udp_bin), _udp_queue,
                    videorate, scale, convert, capsfilter, _udp_flip, encoder, parser, pay, _udp_sink, nullptr);
    if (!gst_element_link_many(_udp_queue,
                    videorate, scale, convert, capsfilter, _udp_flip, encoder, parser, pay, _udp_sink, nullptr))
      throw std::runtime_error("Could not link elements");

}

};
