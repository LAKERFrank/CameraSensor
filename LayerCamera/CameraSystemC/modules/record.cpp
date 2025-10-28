#include "record.h"

using namespace Module;

namespace fs = std::filesystem;

Record::Record(GstElement *main_pipeline, GstElement *main_tee,
               std::shared_ptr<gsttcam::ImageBuffer> imgbuf)
{
    this->_main_bin = main_pipeline;
    this->_main_tee = main_tee;
    this->_imgbuf = imgbuf;

    this->createPipeline();
    this->createH264Bin();
    this->createLosslessBin();
}

void Record::createH264Bin()
{
    if (_file_h264_bin)
    {
        return;
    }

    _file_h264_bin = gst_bin_new("record_h264_bin");

    GstElement *queue = gst_element_factory_make("queue", NULL);
    g_object_set(queue, "leaky", 1, NULL);
    _file_h264_encoder = gst_element_factory_make("nvh264enc", NULL);
    GstElement *parse = gst_element_factory_make("h264parse", NULL);
    GstElement *muxer = gst_element_factory_make("mp4mux", NULL);
    _file_h264_sink = gst_element_factory_make("filesink", "record_h264_sink");

    g_object_set(G_OBJECT(_file_h264_sink), "sync", false, NULL);

    GstPad *sink_pad = gst_element_get_static_pad(_file_h264_sink, "sink");
    gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                      record_eos_callback, this, NULL);
    gst_object_unref(sink_pad);

    gst_bin_add_many(GST_BIN(_file_h264_bin), queue, _file_h264_encoder, parse,
                     muxer, _file_h264_sink, NULL);

    if (!gst_element_link_many(queue, _file_h264_encoder, parse, muxer, _file_h264_sink, NULL))
    {
        throw std::runtime_error("Could not link record (h264 filesink) elements");
    }

    /* add ghostpad */
    GstPad *pad = gst_element_get_static_pad (queue, "sink");
    gst_element_add_pad (_file_h264_bin, gst_ghost_pad_new ("sink", pad));
    gst_object_unref (GST_OBJECT (pad));
}

void Record::createLosslessBin()
{
    if (_file_lossless_bin)
    {
        return;
    }

    _file_lossless_bin = gst_bin_new("record_lossless_bin");

    GstElement *queue = gst_element_factory_make("queue", NULL);
    GstElement *_file_lossless_encoder = gst_element_factory_make("nvh265enc", NULL);
    g_object_set(_file_lossless_encoder, "preset", 7, NULL); // (7): lossless-hp - Lossless, High Performance 
    g_object_set(_file_lossless_encoder, "rc-mode", 1, NULL); // (1): constqp - Constant Quantization
    GstElement *parser = gst_element_factory_make("h265parse", NULL);
    GstElement *muxer = gst_element_factory_make("mp4mux", NULL);
    _file_lossless_sink = gst_element_factory_make("filesink", "record_lossless_sink");
    g_object_set(G_OBJECT(_file_lossless_sink), "sync", false, NULL);

    GstPad *sink_pad = gst_element_get_static_pad(_file_lossless_sink, "sink");
    gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                      record_eos_callback, this, NULL);
    gst_object_unref(sink_pad);

    gst_bin_add_many(GST_BIN(_file_lossless_bin), queue, _file_lossless_encoder, parser,
                     muxer, _file_lossless_sink, NULL);

    if (!gst_element_link_many(queue, _file_lossless_encoder, parser, muxer, _file_lossless_sink, NULL))
    {
        throw std::runtime_error("Could not link record (lossless filesink) elements");
    }

    /* add ghostpad */
    GstPad *pad = gst_element_get_static_pad (queue, "sink");
    gst_element_add_pad (_file_lossless_bin, gst_ghost_pad_new ("sink", pad));
    gst_object_unref (GST_OBJECT (pad));
}

void Record::createPipeline()
{
    if (_record_bin)
    {
        return;
    }

    _record_bin = gst_bin_new("record_bin");

    _record_queue = gst_element_factory_make("queue", "record_queue");
    g_object_set(_record_queue, "leaky", 1, NULL);

    GstElement *record_uploader = gst_element_factory_make("cudaupload", "record_cudaupload");

    GstElement *record_videoconvert = gst_element_factory_make("cudaconvert", "record_videoconvert");

    _record_tee = gst_element_factory_make("tee", "record_tee");

    // for image buffer
    GstElement *imgbuf_queue = gst_element_factory_make("queue", "imgbuf_queue");
    g_object_set(imgbuf_queue, "leaky", 1, NULL);
    //g_object_set(imgbuf_queue, "max-size-buffers", 0, NULL);
    //g_object_set(imgbuf_queue, "max-size-bytes", 0, NULL);
    //g_object_set(imgbuf_queue, "max-size-time", 0, NULL);
    GstElement *imgbuf_convert = gst_element_factory_make("videoconvert", "imgbuf_convert");
    GstElement *imgbuf_scale = gst_element_factory_make("cudascale", "imgbuf_scale");
    GstElement *imgbuf_download = gst_element_factory_make("cudadownload", "imgbuf_cudadownload");
    _imgbuf_capsfilter = gst_element_factory_make("capsfilter", "imgbuf_capsfilter");
    GstElement *imgbuf_sink = gst_element_factory_make("appsink", "imgbuf_sink");
    g_object_set(G_OBJECT(imgbuf_sink), "sync", false, NULL);
    g_object_set(G_OBJECT(imgbuf_sink), "emit-signals", TRUE, NULL);
    g_signal_connect(imgbuf_sink, "new-sample", G_CALLBACK(imgbuf_callback), this);
    g_signal_connect(imgbuf_sink, "eos", G_CALLBACK(imgbuf_eos_callback), this);
    //GstCaps *caps2 = gst_caps_from_string("video/x-raw, width=512, height=288, format=GRAY8");
    //g_object_set(G_OBJECT(_imgbuf_capsfilter), "caps", caps2, nullptr);
    //gst_caps_unref(caps2);

    gst_bin_add_many(GST_BIN(_record_bin), _record_queue, record_uploader,
                     record_videoconvert, _record_tee,
                     imgbuf_download, imgbuf_scale, imgbuf_queue,
                     imgbuf_convert, _imgbuf_capsfilter, imgbuf_sink, nullptr);
    
    // link main pipeline
    if (!gst_element_link_many(
            _record_queue, record_uploader, record_videoconvert, _record_tee,
            imgbuf_queue, imgbuf_scale, imgbuf_download, imgbuf_convert,
            _imgbuf_capsfilter, imgbuf_sink, nullptr))
    {
        throw std::runtime_error("Could not link record (image_queue) elements");
    }

    /* add ghostpad */
    GstPad *pad = gst_element_get_static_pad (_record_queue, "sink");
    gst_element_add_pad (_record_bin, gst_ghost_pad_new ("sink", pad));
    gst_object_unref (GST_OBJECT (pad));
}

void Record::startRecording(std::string filename, bool is_imgbuf_enabled,
                            int imgbuf_width, int imgbuf_height, std::string mode)
{
    if (is_recording) {
        return;
    }

    is_record_eos = false;
    is_imgbuf_eos = false;
    g_atomic_int_set(&(this->frame_idx), 0);
    _enable_image_buf = is_imgbuf_enabled;

    if (is_imgbuf_enabled) {
        this->_imgbuf->clear();

        // set image buffer output (width, height)
        GstCaps *caps = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "GRAY8",
            "width", G_TYPE_INT, imgbuf_width,
            "height", G_TYPE_INT, imgbuf_height,
            nullptr);
        g_object_set(G_OBJECT(_imgbuf_capsfilter), "caps", caps, nullptr);
        gst_caps_unref(caps);
    }
    else {
        g_object_set(_imgbuf_capsfilter, "caps", NULL, NULL);
    }

    fs::path basepath = filename;
    basepath.replace_extension();

    // convert parameters
    if (mode == "lossless") { this->mode = RecordMode::LOSSLESS;  }
    else if (mode == "h264_low") { this->mode = RecordMode::H264_LOW; }
    else if (mode == "h264_high") { this->mode = RecordMode::H264_HIGH; }
    else if (mode == "none" || mode == "") { this->mode = RecordMode::NONE; }
    else {
        LOG_RECORD_MODULE << "ERROR: mode=" << mode << " not acceptable, fallback to default (none)" << std::endl;
        this->mode = RecordMode::NONE;
    }

    if (this->mode != RecordMode::NONE)
    {
        meta_csv_.open((basepath.string() + "_meta.csv").c_str());
        meta_csv_ << "index,timestamp,monotonic_timestamp\n";
    }

    switch (this->mode) {

    case RecordMode::LOSSLESS:
        g_object_set(_file_lossless_sink, "location", (basepath.string() + ".mp4").c_str(), NULL);
        gst_element_set_state(_file_lossless_bin, GST_STATE_READY);
        gst_bin_add(GST_BIN(_record_bin), _file_lossless_bin);
        if (!gst_element_link(_record_tee, _file_lossless_bin)) {
            throw std::runtime_error("Could not link record_bin with lossless_bin");
        }
        break;
    case RecordMode::H264_LOW:
    case RecordMode::H264_HIGH:
        if (this->mode == RecordMode::H264_LOW) {
            g_object_set(_file_h264_encoder, "rc-mode", 2, "bitrate", 2000, NULL);
        }
        else {
            g_object_set(_file_h264_encoder, "rc-mode", 7, "bitrate", 10000, NULL);
        }
        g_object_set(_file_h264_sink, "location", (basepath.string() + ".mp4").c_str(), NULL);
        gst_element_set_state(_file_h264_bin, GST_STATE_READY);
        gst_bin_add(GST_BIN(_record_bin), _file_h264_bin);
        if (!gst_element_link(_record_tee, _file_h264_bin)) {
            throw std::runtime_error("Could not link record_bin with h264_bin");
        }
        break;

    case RecordMode::NONE:
    default:
        break;

    }

    // Start playing
    gst_element_set_state(_record_bin, GST_STATE_PLAYING);

    gst_bin_add(GST_BIN(_main_bin), _record_bin);

    if (!gst_element_link(_main_tee, _record_bin))
    {
        std::cerr << "ERROR linking elements" << std::endl;
    }

    //gst_element_set_state(_main_bin, GST_STATE_PLAYING);

    LOG_RECORD_MODULE << "Start recording" << std::endl;

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_main_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[record_module]start(main)");

    is_recording = true;
}

void Record::stopRecording()
{
    if (!is_recording) {
        return;
    }
    int c;

    gst_element_unlink(_main_tee, _record_bin);
    gst_bin_remove(GST_BIN(_main_bin), _record_bin);

    // set main pipeline to PLAYING state
    //c = gst_element_set_state(_main_bin, GST_STATE_PLAYING);
    //LOG_RECORD_MODULE << "state change (set main_bin to PLAYING state) = " << c << std::endl;

    // send EOS event
    gst_element_send_event(_record_bin, gst_event_new_eos());

    // set record pipeline to PLAYING state
    c = gst_element_set_state(_record_bin, GST_STATE_PLAYING);
    LOG_RECORD_MODULE << "state change (set record_bin to PLAYING state) = " << c << std::endl;
    gst_element_get_state(_record_bin, NULL, NULL, GST_CLOCK_TIME_NONE);

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_main_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[record_module]stop(main)");

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_record_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[record_module]stop(record)");

    LOG_RECORD_MODULE << "Stopped recording. Waiting for EOS." << std::endl;

    // waiting for final element reach EOS event.
    std::unique_lock<std::mutex> lock(eos_mutex);
    if (this->mode == RecordMode::NONE) {
        eos_cond.wait(lock, [&] { return is_imgbuf_eos; });
    }
    else {
        eos_cond.wait(lock, [&] { return is_record_eos && is_imgbuf_eos; });
    }

    // HOTFIX:
    // 因為接收到上述 is_record_eos 時，filesink有機率還沒把資料處理完成
    // 導致 mp4 寫入不完全無法撥放，所以這邊先加一個 hotfix 睡1秒
    // 後續可能要用 bus_watch 等方法確認整個record_bin已經把 EOS event 處理完才能關閉
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // stop record pipeline
    c = gst_element_set_state(_record_bin, GST_STATE_NULL);
    LOG_RECORD_MODULE << "state change (set record_bin to NULL state) = " << c << std::endl;
    // ensuer state has changed
    gst_element_get_state(_record_bin, NULL, NULL, GST_CLOCK_TIME_NONE);

    switch (this->mode) {

    case RecordMode::LOSSLESS:
        gst_element_unlink(_record_tee, _file_lossless_bin);
        if (!gst_bin_remove(GST_BIN(_record_bin), _file_lossless_bin)) {
            throw std::runtime_error("Could not remove lossless_bin from record_bin");
        }
        break;
    case RecordMode::H264_LOW:
    case RecordMode::H264_HIGH:
        gst_element_unlink(_record_tee, _file_h264_bin);
        if (!gst_bin_remove(GST_BIN(_record_bin), _file_h264_bin)) {
            throw std::runtime_error("Could not remove h264_bin from record_bin");
        }
        break;
    case RecordMode::NONE:

    default:
        break;

    }

    // unlink proxy pad
    //GstPad *sink_pad = gst_element_get_static_pad(_record_queue, "sink");
    //GstPad *proxy_pad = gst_pad_get_peer(sink_pad);
    //gst_pad_unlink(proxy_pad, sink_pad);
    //gst_object_unref(proxy_pad);
    //gst_object_unref(sink_pad);

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_main_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[record_module]stop_final(main)");
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_record_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[record_module]stop_final(record)");
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_file_h264_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[record_module]stop_final(h264)");
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_file_lossless_bin), GST_DEBUG_GRAPH_SHOW_ALL, "[record_module]stop_final(lossless)");

    reset();

    LOG_RECORD_MODULE << "Stopped recording...ok" << std::endl;
}

void Record::reset()
{
    is_recording = false;
    is_imgbuf_eos = false;
    is_record_eos = false;
}

GstPadProbeReturn Record::record_eos_callback(GstPad *,
                                              GstPadProbeInfo *info,
                                              gpointer user_data)
{
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM)
    {
        GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
        if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
        {

            Record *rec = static_cast<Record *>(user_data);

            rec->is_record_eos = true;
            rec->eos_cond.notify_all();

            LOG_RECORD_MODULE << "[filesink] EOS reached the end with tid = " << gettid() << "\n";
        }
    }
    return GST_PAD_PROBE_OK;
}

GstFlowReturn Record::imgbuf_callback(GstElement *sink, gpointer user_data)
{
    Record *rec = static_cast<Record *>(user_data);

    GstSample *sample = NULL;
    /* Retrieve the buffer */
    g_signal_emit_by_name(sink, "pull-sample", &sample, NULL);

    std::shared_ptr<gsttcam::Frame> frame = std::make_shared<gsttcam::Frame>();

    if (sample)
    {
        GstBuffer *buffer = gst_sample_get_buffer(sample);

        GstMeta *meta = gst_buffer_get_meta(buffer, g_type_from_name("TcamStatisticsMetaApi"));

        if (meta)
        {
            GstStructure *struc = ((TcamStatisticsMeta *)meta)->structure;
            guint64 nsec;
            gst_structure_get(struc, "capture_time_ns", G_TYPE_UINT64, &nsec, NULL);
            frame->monotonic_timestamp = nsec / 1000000000.0;
            frame->timestamp = (nsec + rec->clock_offset) / 1000000000.0;
        }
        else
        {
            g_warning("No meta data available");
        }

        frame->index = g_atomic_int_get(&rec->frame_idx);
        g_atomic_int_inc(&rec->frame_idx);

        if (rec->mode != RecordMode::NONE) {
            rec->meta_csv_ << frame->index << "," << std::fixed << std::setprecision(6) << frame->timestamp << "," << frame->monotonic_timestamp << "\n";
        }

        if (rec->_enable_image_buf)
        {
            GstCaps *caps = gst_sample_get_caps(sample);

            GstStructure *caps_struct = gst_caps_get_structure(caps, 0);
            if (!caps_struct)
            {
                g_warning("caps have NULL structure");
            }

            if (!gst_structure_get_int(caps_struct, "width", &frame->width) ||
                !gst_structure_get_int(caps_struct, "height", &frame->height))
            {
                g_warning("caps have no HEIGHT, WIDTH");
            }

            GstMapInfo info;
            gst_buffer_map(buffer, &info, GST_MAP_READ);
            gst_buffer_unmap(buffer, &info);
            frame->memory_size = info.size;

            std::shared_ptr<u_int8_t> memory((u_int8_t *)malloc(info.size), free);
            memcpy(memory.get(), info.data, info.size);
            frame->memory = memory;
            //LOG_RECORD_MODULE << "ImageBuffer push frame id=" << frame->index << std::endl;
            rec->_imgbuf->push(frame);
        }

        // delete our reference so that gstreamer can handle the sample
        gst_sample_unref(sample);
    }
    return GST_FLOW_OK;
}

void Record::imgbuf_eos_callback(GstElement *, gpointer user_data)
{
    Record *rec = static_cast<Record *>(user_data);

    if (rec->_enable_image_buf)
    {
        std::shared_ptr<gsttcam::Frame> frame = std::make_shared<gsttcam::Frame>();
        frame->is_eos = true;

        rec->_imgbuf->push(frame);
    }

    if (rec->mode != RecordMode::NONE) {
        rec->meta_csv_.close();
    }

    rec->is_imgbuf_eos = true;
    rec->eos_cond.notify_all();

    LOG_RECORD_MODULE << "[imgbuf] EOS reached the end with tid = " << gettid() << "\n";
}

void Record::setClockOffset(long long v) {
    this->clock_offset = v;
}
