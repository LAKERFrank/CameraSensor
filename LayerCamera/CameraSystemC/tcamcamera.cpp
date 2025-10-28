#include "tcamcamera.h"

#include <unistd.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// #include <QDebug>
#include <assert.h>
#include <gst/app/gstappsink.h>

#include "gstmetatcamstatistics.h"

using namespace gsttcam;

std::vector<CameraInfo> gsttcam::get_device_list() {
  if (!gst_is_initialized())
    throw std::runtime_error(
        "GStreamer is not initialized! gst_init(...) "
        "needs to be called before using this function.");

  // The device monitor listens to device activities for us
  GstDeviceMonitor *monitor = gst_device_monitor_new();
  // We are only interested in devices that are in the categories
  // Video and Source && tcam
  gst_device_monitor_add_filter(monitor, "Video/Source/tcam", NULL);

  //
  // static query
  // list all devices that are available right now
  //

  GList *devices = gst_device_monitor_get_devices(monitor);

  std::vector<CameraInfo> ret;
  for (GList *elem = devices; elem; elem = elem->next) {
    GstDevice *device = (GstDevice *)elem->data;


    GstStructure *struc = gst_device_get_properties(device);

    if (std::string(gst_structure_get_string(struc, "type")) == std::string("v4l2")) {
      CameraInfo info;

      info.serial = gst_structure_get_string(struc, "serial");
      info.name = gst_structure_get_string(struc, "model");

      ret.push_back(info);
    }

    gst_structure_free(struc);
  }

  g_list_free_full(devices, gst_object_unref);

  return ret;
}

Property::Property(TcamPropertyBase *baseProperty, std::string name) {
  this->baseProperty = baseProperty;
  this->name = name;
  this->displayName = tcam_property_base_get_display_name(baseProperty);
  this->category = tcam_property_base_get_category(baseProperty);
  this->description = tcam_property_base_get_description(baseProperty);

  GError *err = NULL;
  this->available = tcam_property_base_is_available(baseProperty, &err);
  this->locked = tcam_property_base_is_available(baseProperty, &err);
};
Property::~Property() { g_object_unref(baseProperty); };

IntegerProperty::IntegerProperty(TcamPropertyBase *base, std::string name)
    : Property(base, name) {
  this->type = "integer";
  this->integerProperty = TCAM_PROPERTY_INTEGER(base);

  GError *err = NULL;

  value = tcam_property_integer_get_value(integerProperty, &err);

  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  default_value =
      tcam_property_integer_get_default(this->integerProperty, &err);
  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  tcam_property_integer_get_range(this->integerProperty, &min, &max, &step,
                                  &err);

  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  const char *tmp_unit = tcam_property_integer_get_unit(integerProperty);

  if (tmp_unit) {
    this->unit = std::string(tmp_unit);
  }
};

bool IntegerProperty::set(int value) {
  GError *err = NULL;
  tcam_property_integer_set_value(integerProperty, value, &err);

  if (err) {
    printf("Error while setting property: %s\n", err->message);
    g_error_free(err);
    err = NULL;
    return false;
  } else {
    //printf("Set %s to %d\n", name.c_str(), value);
    return true;
  }
  return true;
};
DoubleProperty::DoubleProperty(TcamPropertyBase *base, std::string name)
    : Property(base, name) {
  this->type = "double";
  this->doubleProperty = TCAM_PROPERTY_FLOAT(base);

  GError *err = NULL;

  value = tcam_property_float_get_value(doubleProperty, &err);
  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  default_value = tcam_property_float_get_default(this->doubleProperty, &err);
  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  tcam_property_float_get_range(this->doubleProperty, &min, &max, &step, &err);

  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  const char *tmp_unit = tcam_property_float_get_unit(doubleProperty);

  if (tmp_unit) {
    this->unit = std::string(tmp_unit);
  }
};

bool DoubleProperty::set(double value) {
  GError *err = NULL;
  tcam_property_float_set_value(doubleProperty, value, &err);

  if (err) {
    printf("Error while setting property: %s\n", err->message);
    g_error_free(err);
    err = NULL;
    return false;
  } else {
    //printf("Set %s to %f\n", name.c_str(), value);
    return true;
  }
  return true;
};

EnumProperty::EnumProperty(TcamPropertyBase *base, std::string name)
    : Property(base, name) {
  this->type = "enum";
  this->enumProperty = TCAM_PROPERTY_ENUMERATION(base);

  GError *err = NULL;

  value = tcam_property_enumeration_get_value(this->enumProperty, &err);
  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  default_value =
      tcam_property_enumeration_get_default(this->enumProperty, &err);
  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  GSList *enum_entries =
      tcam_property_enumeration_get_enum_entries(this->enumProperty, &err);

  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  if (enum_entries) {
    for (GSList *entry = enum_entries; entry != NULL; entry = entry->next) {
      this->values.push_back((const char *)entry->data);
    }

    g_slist_free_full(enum_entries, g_free);
  }
};

bool EnumProperty::set(std::string value) {
  GError *err = NULL;
  tcam_property_enumeration_set_value(this->enumProperty, value.c_str(), &err);

  if (err) {
    printf("Error while setting property: %s\n", err->message);
    g_error_free(err);
    err = NULL;
    return false;
  } else {
    //printf("Set %s to %s\n", name.c_str(), value.c_str());
    return true;
  }
  return true;
};

BooleanProperty::BooleanProperty(TcamPropertyBase *base, std::string name)
    : Property(base, name) {
  this->type = "boolean";
  this->booleanProperty = TCAM_PROPERTY_BOOLEAN(base);

  GError *err = NULL;

  value = tcam_property_boolean_get_value(booleanProperty, &err);
  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }

  default_value =
      tcam_property_boolean_get_default(this->booleanProperty, &err);
  if (err) {
    printf("%s\n", err->message);
    g_error_free(err);
    err = NULL;
  }
};

bool BooleanProperty::set(bool value) {
  GError *err = NULL;
  tcam_property_boolean_set_value(booleanProperty, value, &err);

  if (err) {
    printf("Error while setting property: %s\n", err->message);
    g_error_free(err);
    err = NULL;
    return false;
  } else {
    //printf("Set %s to %s\n", name.c_str(), value ? "true" : "false");
    return true;
  }
  return true;
};

std::string VideoFormatCaps::to_string() {
  std::string ret;
  if (size.width != 0) {
    ret += "size = " + std::to_string(size.width) + " x " +
           std::to_string(size.height) + " ";
  } else {
    ret += "size range = [" + std::to_string(size_min.width) + " x ";
    ret += std::to_string(size_min.height) + ", ";
    ret += std::to_string(size_max.width) + " x " +
           std::to_string(size_max.height) + "] ";
  }
  ret += "color formats = {";
  for (std::string &colorfmt : formats) {
    ret += colorfmt + ",";
  }
  ret += "} ";
  if (!framerates.empty()) {
    ret += "frame rates = {";
    for (FrameRate &rate : framerates) {
      ret += std::to_string(rate.numerator) + "/" +
             std::to_string(rate.denominator) + ",";
    }
    ret += "}";
  } else {
    ret += "frame rate range = [";
    ret += std::to_string(framerate_min.numerator) + "/";
    ret += std::to_string(framerate_min.denominator) + ",";
    ret += std::to_string(framerate_max.numerator) + "/";
    ret += std::to_string(framerate_max.denominator) + "]";
  }

  return ret;
}

TcamCamera::TcamCamera(std::string serial, std::shared_ptr<ImageBuffer> image_queue,
                       std::shared_ptr<MetricData> metric_data, bool clockoverlay,
                       bool enable_software_trigger, bool enable_resync)
{
  serial_ = serial;
  if (!gst_is_initialized())
    throw std::runtime_error(
        "GStreamer is not initialized! gst_init(...) "
        "needs to be called before using this function.");
  if (serial == std::string("None")) {
    create_fake_pipeline();
  } else {
    create_pipeline(clockoverlay);
  }
  if (serial != "") g_object_set(tcambin_, "serial", serial.c_str(), nullptr);

  if (!is_fake) {
    videocaps_ = initialize_format_list();
  }

  this->image_queue = image_queue;

  this->initClockOffset();

  this->software_trigger = enable_software_trigger;
  this->enable_resync = enable_resync;

  //LOGMYSYSTEM << "TcamCamera::software_trigger=" << (enable_software_trigger ? "true" : "false") << std::endl;
  //LOGMYSYSTEM << "TcamCamera::enable_resync=" << (enable_resync ? "true" : "false") << std::endl;

  this->module_monitor = new Module::Monitor(pipeline_, tee_, metric_data);
  this->module_monitor->setClockOffset(this->clock_offset_ns);
}

TcamCamera::~TcamCamera() {
  // remove monitor
  delete this->module_monitor;
  this->module_monitor = nullptr;

  LOGMYSYSTEM << "pipeline refcount at cleanup: "
              << GST_OBJECT_REFCOUNT_VALUE(pipeline_) << "\n";
  gst_object_unref(pipeline_);
}

void TcamCamera::create_fake_pipeline() {
  is_fake = true;
  pipeline_ = gst_pipeline_new("pipeline");
  tcambin_ = gst_element_factory_make("videotestsrc", nullptr);
  if (!tcambin_)
    throw std::runtime_error(
        "'videotestsrc' could not be initialized! Check tiscamera "
        "installation");
  g_object_set(tcambin_, "is-live", true, nullptr);
  GstElement *conv = gst_element_factory_make("videoconvert", nullptr);
  capturecapsfilter_ = gst_element_factory_make("capsfilter", nullptr);
  tee_ = gst_element_factory_make("tee", "tee");
  GstElement *queue = gst_element_factory_make("queue", nullptr);
  capturesink_ = gst_element_factory_make("fakesink", nullptr);
  // g_object_set(capturesink_, "max-buffers", 4, "drop", true, nullptr);
  assert(pipeline_ && tee_ && capturecapsfilter_ && queue && capturesink_);

  gst_bin_add_many(GST_BIN(pipeline_), tcambin_, conv, capturecapsfilter_, tee_,
                   queue, capturesink_, nullptr);
  const auto ret = gst_element_link_many(tcambin_, conv, capturecapsfilter_,
                                         tee_, queue, capturesink_, nullptr);
  assert(ret);
  if (!ret) throw std::runtime_error("Unable to link pipeline");
}

void TcamCamera::create_pipeline(bool is_clockoverlay) {
  pipeline_ = gst_pipeline_new("pipeline");
  tcambin_ = gst_element_factory_make("tcambin", nullptr);
  if (!tcambin_)
    throw std::runtime_error(
        "'tcambin' could not be initialized! Check tiscamera installation");
  GstElement *conv = gst_element_factory_make("videoconvert", nullptr);
  capturecapsfilter_ = gst_element_factory_make("capsfilter", nullptr);
  tee_ = gst_element_factory_make("tee", "tee");
  g_object_set(G_OBJECT(tee_), "allow-not-linked", true, nullptr);

  assert(pipeline_ && tee_ && capturecapsfilter_);

  gst_bin_add_many(GST_BIN(pipeline_), tcambin_, conv, capturecapsfilter_, tee_, nullptr);

  bool ret;
  if (is_clockoverlay) {
    GstElement *clockoverlay = gst_element_factory_make("clockoverlay", nullptr);
    g_object_set(G_OBJECT(clockoverlay), "time-format", "%Y-%m-%d %H:%M:%S", nullptr);
    gst_bin_add(GST_BIN(pipeline_), clockoverlay);
    ret = gst_element_link_many(tcambin_, conv, capturecapsfilter_, clockoverlay,
                                           tee_, nullptr);
  }
  else {
    ret = gst_element_link_many(tcambin_, conv, capturecapsfilter_,
                                           tee_, nullptr);
  }
  if (!ret) throw std::runtime_error("Unable to link source pipeline");
}

void TcamCamera::ensure_ready_state() {
  GstState state;
  if ((gst_element_get_state(tcambin_, &state, nullptr, GST_CLOCK_TIME_NONE) ==
       GST_STATE_CHANGE_SUCCESS) &&
      state == GST_STATE_NULL) {
    gst_element_set_state(tcambin_, GST_STATE_READY);
    gst_element_get_state(tcambin_, nullptr, nullptr, GST_CLOCK_TIME_NONE);
  }
}

std::vector<VideoFormatCaps> TcamCamera::initialize_format_list() {
  gst_element_set_state(tcambin_, GST_STATE_READY);
  gst_element_get_state(tcambin_, nullptr, nullptr, GST_CLOCK_TIME_NONE);

  std::vector<VideoFormatCaps> ret;

  GstPad *pad = gst_element_get_static_pad(tcambin_, "src");
  assert(pad);
  GstCaps *caps = gst_pad_query_caps(pad, nullptr);
  assert(caps);

  for (guint i = 0; i < gst_caps_get_size(caps); ++i) {
    VideoFormatCaps fmt = {0};
    GstStructure *s = gst_caps_get_structure(caps, i);
    if (s) {
      if (!g_strcmp0(gst_structure_get_name(s), "ANY")) continue;
      const GValue *width = gst_structure_get_value(s, "width");
      const GValue *height = gst_structure_get_value(s, "height");
      assert(width && height);
      if (G_VALUE_HOLDS_INT(width)) {
        assert(G_VALUE_HOLDS_INT(height));
        fmt.size.width = g_value_get_int(width);
        fmt.size.height = g_value_get_int(height);
      } else if (GST_VALUE_HOLDS_INT_RANGE(width)) {
        assert(GST_VALUE_HOLDS_INT_RANGE(height));
        fmt.size_min.width = gst_value_get_int_range_min(width);
        fmt.size_min.height = gst_value_get_int_range_min(height);
        fmt.size_max.width = gst_value_get_int_range_max(width);
        fmt.size_max.height = gst_value_get_int_range_max(height);
      } else {
        assert(FALSE && "Invalid or missing width/height");
      }

      // Handle color formats
      const GValue *format = gst_structure_get_value(s, "format");
      // TODO: Support jpeg
      if (format == nullptr) continue;
      if (G_VALUE_HOLDS_STRING(format)) {
        fmt.formats.push_back(
            std::string(gst_structure_get_string(s, "format")));
      } else if (GST_VALUE_HOLDS_LIST(format)) {
        for (guint i = 0; i < gst_value_list_get_size(format); i++) {
          const GValue *val = gst_value_list_get_value(format, i);
          assert(val && G_VALUE_HOLDS_STRING(val));
          fmt.formats.push_back(std::string(g_value_get_string(val)));
        }
      } else {
        assert(FALSE && "Invalid or missing format");
      }

      // Handle frame rates
      const GValue *framerates = gst_structure_get_value(s, "framerate");
      assert(framerates);
      if (GST_VALUE_HOLDS_LIST(framerates)) {
        for (guint i = 0; i < gst_value_list_get_size(framerates); i++) {
          const GValue *val = gst_value_list_get_value(framerates, i);
          assert(val && GST_VALUE_HOLDS_FRACTION(val));
          FrameRate rate;
          rate.numerator = gst_value_get_fraction_numerator(val);
          rate.denominator = gst_value_get_fraction_denominator(val);
          fmt.framerates.push_back(rate);
        }
      } else if (GST_VALUE_HOLDS_FRACTION_RANGE(framerates)) {
        const GValue *min = gst_value_get_fraction_range_min(framerates);
        const GValue *max = gst_value_get_fraction_range_max(framerates);
        assert(min && max);
        fmt.framerate_min.numerator = gst_value_get_fraction_numerator(min);
        fmt.framerate_min.denominator = gst_value_get_fraction_denominator(min);
        fmt.framerate_max.numerator = gst_value_get_fraction_numerator(max);
        fmt.framerate_max.denominator = gst_value_get_fraction_denominator(max);
      } else {
        assert(FALSE && "Invalid or missing framerate");
      }

      ret.push_back(fmt);
    }
  }
  gst_caps_unref(caps);
  g_object_unref(pad);

  return ret;
}

std::vector<VideoFormatCaps> TcamCamera::get_format_list() {
  return videocaps_;
}

std::shared_ptr<Property> TcamCamera::get_property(std::string name) {
  GError *err = NULL;
  TcamPropertyBase *base_property = tcam_property_provider_get_tcam_property(
      TCAM_PROPERTY_PROVIDER(tcambin_), name.c_str(), &err);
  TcamPropertyType type = tcam_property_base_get_property_type(base_property);

  std::shared_ptr<Property> prop;

  switch (type) {
    case TCAM_PROPERTY_TYPE_INTEGER: {
      prop =
          std::shared_ptr<Property>(new IntegerProperty(base_property, name));
      break;
    }
    case TCAM_PROPERTY_TYPE_FLOAT: {
      prop = std::shared_ptr<Property>(new DoubleProperty(base_property, name));
      break;
    }
    case TCAM_PROPERTY_TYPE_BOOLEAN: {
      prop =
          std::shared_ptr<Property>(new BooleanProperty(base_property, name));
      break;
    }
    case TCAM_PROPERTY_TYPE_ENUMERATION: {
      prop = std::shared_ptr<Property>(new EnumProperty(base_property, name));
      break;
    }
    default: {
      prop = std::shared_ptr<Property>(new Property(base_property, name));
      break;
    }
  }

  return prop;
}

void TcamCamera::setProperty(std::string name, std::string s_value) {
  std::shared_ptr<Property> prop = this->get_property(name);
  if (prop->type == "integer") {
    std::shared_ptr<gsttcam::IntegerProperty> intprop =
        std::dynamic_pointer_cast<gsttcam::IntegerProperty>(prop);
    intprop->set(std::stoi(s_value));
  } else if (prop->type == "double") {
    std::shared_ptr<gsttcam::DoubleProperty> doubleprop =
        std::dynamic_pointer_cast<gsttcam::DoubleProperty>(prop);
    doubleprop->set(std::stod(s_value));
  } else if (prop->type == "enum") {
    std::shared_ptr<gsttcam::EnumProperty> enumprop =
        std::dynamic_pointer_cast<gsttcam::EnumProperty>(prop);
    enumprop->set(s_value);
  } else if (prop->type == "boolean") {
    std::shared_ptr<gsttcam::BooleanProperty> boolprop =
        std::dynamic_pointer_cast<gsttcam::BooleanProperty>(prop);
    boolprop->set(s_value == std::string("On"));
  }
}

std::vector<std::shared_ptr<Property>> TcamCamera::get_camera_property_list() {
  std::vector<std::shared_ptr<Property>> pptylist;

  if (is_fake) {
    return pptylist;
  }

  GError *err = NULL;
  GSList *names = tcam_property_provider_get_tcam_property_names(
      TCAM_PROPERTY_PROVIDER(tcambin_), &err);

  for (unsigned int i = 0; i < g_slist_length(names); ++i) {
    char *name = (char *)g_slist_nth(names, i)->data;
    try {
      std::shared_ptr<Property> prop = get_property(std::string(name));
      pptylist.push_back(prop);
    } catch (...) {
    }
  }
  return pptylist;
}

void TcamCamera::set_capture_format(FrameSize size,
                                    FrameRate framerate, std::string skipping) {
  // set sleep trigger
  trigger_nsec = (int)(1e9 * framerate.denominator / framerate.numerator);
  LOGMYSYSTEM << "trigger_nsec = " << trigger_nsec << std::endl;
  // Override fps with highest fps (to make gstreamer function normally)
  if (this->software_trigger) {

    if (size.width == 1440 && size.height == 1080) {
      framerate = {2500000, 10593}; // 236 fps
    }
    else if (size.width == 640 && size.height == 480 && skipping == "2x2") {
      framerate = {5000000, 20833}; // 240 fps
    }
    else if (size.width == 2048 && size.height == 1536) {
      framerate = {238, 1};
    }
    else if (size.width == 1024 && size.height == 768 && skipping == "2x2") {
      framerate = {312500, 1313}; // 238 fps
    }
  }

  LOGMYSYSTEM << "set_capture_format {size: (" << size.width << ","
              << size.height << "), framerate=" << framerate.numerator << "/"
              << framerate.denominator;

  if (!skipping.empty()) {
    std::cout << ", skipping=" << skipping;
  }
  std::cout << "}" << std::endl;

  GstCaps *caps = gst_caps_new_simple(
      "video/x-raw",
      "format", G_TYPE_STRING, "BGRx",
      "width", G_TYPE_INT, size.width,
      "height", G_TYPE_INT, size.height,
      "framerate", GST_TYPE_FRACTION, framerate.numerator, framerate.denominator,
      "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
      nullptr);
  assert(caps);

  if (!skipping.empty()) {
    gst_caps_set_simple(caps, "skipping", G_TYPE_STRING, skipping.c_str(), NULL);
  }

  g_object_set(G_OBJECT(capturecapsfilter_), "caps", caps, nullptr);
  gst_caps_unref(caps);
}

bool TcamCamera::start(unsigned long long trigger_start) {
  GError *err = NULL;
  LOGMYSYSTEM << "start() ..." << std::endl;
  if (this->software_trigger) {
    LOGMYSYSTEM << "Software Trigger Setting ..." << std::endl;
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerSelector", "Frame Start", &err);
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerActivation", "Rising Edge", &err);
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerOperation", "Default", &err);
    //tcam_property_provider_set_tcam_float(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerDelay", 0.0f, &err);
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerMode", "On", &err);

    if (trigger_start != 0) {
      LOGMYSYSTEM << "trigger start set." << std::endl;
      this->trigger_tp = std::chrono::system_clock::time_point {
        std::chrono::nanoseconds(trigger_start)
      };
    }
    else {
      this->trigger_tp = std::chrono::system_clock::now();
    }

    thread_running = true;
    trigger_thread = std::thread(&TcamCamera::trigger_func, this);

    // setting thread name
    std::string name = "s_trigger";
    pthread_setname_np(trigger_thread.native_handle(), name.substr(0, 15).c_str()); 

    // Set real-time scheduling policy and priority
    sched_param sch_params;
    sch_params.sched_priority = 80; // Priority 1–99 for real-time

    if (pthread_setschedparam(trigger_thread.native_handle(), SCHED_RR, &sch_params)) {
        std::cerr << "Failed to set thread realtime priority: "
                  << strerror(errno) << '\n';
    } else {
        std::cout << "Set thread priority to " << sch_params.sched_priority << '\n';
    }
  }
  else {
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerMode", "Off", &err);
  }

  gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  gst_element_get_state(pipeline_, NULL, NULL, GST_CLOCK_TIME_NONE);
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline_), GST_DEBUG_GRAPH_SHOW_ALL,
                            "pipeline");

  if (this->enable_resync) {
    resync_thread = std::thread(&TcamCamera::resync_daemon, this);

    // setting thread name
    std::string name = "resync_trigger";
    pthread_setname_np(resync_thread.native_handle(), name.substr(0, 15).c_str()); 

    // Set real-time scheduling policy and priority
    sched_param sch_params;
    sch_params.sched_priority = 80; // Priority 1–99 for real-time

    if (pthread_setschedparam(resync_thread.native_handle(), SCHED_RR, &sch_params)) {
        std::cerr << "Failed to set thread realtime priority: "
                  << strerror(errno) << '\n';
    } else {
        std::cout << "Set thread priority to " << sch_params.sched_priority << '\n';
    }
    // initialize resync offset
    //this->resync(trigger_start);
  }
  return TRUE;
}

void TcamCamera::resync(unsigned long long t, unsigned long long dt) {
  if (resync_mutex.try_lock()) {
    this->resync_next_timestamp = t;
    this->resync_next_dt = dt;
    resync_mutex.unlock();
    resync_cond.notify_one();
  }
  else {
    LOGMYSYSTEM << "Another resync is running, skipped." << std::endl;
  }
}

bool TcamCamera::stop() {
  disable_video_record();
  disable_loop_record();
  disable_video_snapshot();
  disable_video_udp_stream();
  disable_video_display();
  if (this->software_trigger) {
    GError *err = NULL;
    this->thread_running = false;
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerMode", "Off", &err);
    trigger_thread.join();
  }

  if (this->enable_resync) {
    this->is_resync_daemon_running = false;
    resync_cond.notify_all();
    resync_thread.join();
  }

  gst_element_set_state(pipeline_, GST_STATE_NULL);
  gst_element_get_state(pipeline_, NULL, NULL, GST_CLOCK_TIME_NONE);
  return TRUE;
}

void TcamCamera::enable_video_udp_stream(std::string host, int port) {
  if (!this->module_udp) {
    this->module_udp = new Module::Udp(pipeline_, tee_, host, port, direction);
  }
}

void TcamCamera::disable_video_udp_stream() {
  if (this->module_udp) {
    delete this->module_udp;
    this->module_udp = nullptr;
  }
}

void TcamCamera::enable_video_display(unsigned long win_id) {
  if (!this->module_display) {
    this->module_display =
        new Module::Display(pipeline_, tee_, win_id, direction);
  }
}

void TcamCamera::disable_video_display() {
  if (this->module_display) {
    delete this->module_display;
    this->module_display = nullptr;
  }
}

void TcamCamera::enable_video_record() {
  if (!this->module_record) {
    this->module_record = new Module::Record(pipeline_, tee_, image_queue);
  }
}

void TcamCamera::disable_video_record() {
  if (this->module_record) {
    delete this->module_record;
    this->module_record = nullptr;
  }
}

void TcamCamera::enable_loop_record() {
  if (!this->module_loop_record) {
    this->module_loop_record =
        new Module::LoopRecord(pipeline_, tee_, 640, 480, NULL, 2);
  }
}

void TcamCamera::disable_loop_record() {
  if (this->module_loop_record) {
    delete this->module_loop_record;
    this->module_loop_record = nullptr;
  }
}
#include <chrono>
using namespace std::literals;
void TcamCamera::start_recording(std::string filename, bool enable_image_buf,
                                 int imgbuf_width, int imgbuf_height, std::string mode) {
  this->module_record->setClockOffset(this->clock_offset_ns);
  this->module_record->startRecording(filename.c_str(), enable_image_buf,
                                      imgbuf_width, imgbuf_height, mode);
}

void TcamCamera::stop_recording() { this->module_record->stopRecording(); }

void TcamCamera::startLoopRecording() {
  this->module_loop_record->startLoopRecording();
}

void TcamCamera::stopLoopRecording() {
  this->module_loop_record->stopLoopRecording();
}

void TcamCamera::enable_video_snapshot() {
  if (!this->module_snapshot) {
    this->module_snapshot = new Module::Snapshot(pipeline_, tee_, direction);
  }
}

void TcamCamera::disable_video_snapshot() {
  if (this->module_snapshot) {
    delete this->module_snapshot;
    this->module_snapshot = nullptr;
  }
}

std::shared_ptr<Snapshot> TcamCamera::take_snapshot() {
  if (this->module_snapshot) {
    return this->module_snapshot->take();
  }
  return nullptr;
}

void TcamCamera::set_direction(int direction) { this->direction = direction; }

int TcamCamera::get_direction() { return this->direction; }

void TcamCamera::initClockOffset() {
  // Get current real time
  timespec realtime1 = get_clock_time(CLOCK_REALTIME);
  // Get boot time (real time - uptime)
  timespec boottime = get_clock_time(CLOCK_BOOTTIME);
  // Get current real time
  timespec realtime2 = get_clock_time(CLOCK_REALTIME);

  long long r1 = timespec_to_ns(realtime1);
  long long r2 = timespec_to_ns(realtime2);

  LOGMYSYSTEM;
  printf("check1:%10lld.%9lld\n", (long long)(r1 / 1e9),
         r1 - (long long)((long long)(r1 / 1e9) * 1e9));
  LOGMYSYSTEM;
  printf("check2:%10lld.%9lld\n", (long long)(r2 / 1e9),
         r2 - (long long)((long long)(r2 / 1e9) * 1e9));

  long long diff = r2 - r1;
  LOGMYSYSTEM;
  printf("diff:%10lld.%9lld\n", (long long)(diff / 1e9),
         diff - (long long)((long long)(diff / 1e9) * 1e9));

  this->clock_offset_ns = r1 - timespec_to_ns(boottime);
}

void TcamCamera::resync_daemon() {

  // 持續執行 resync 守護執行緒，直到 is_resync_daemon_running 被設為 false
  while (is_resync_daemon_running) {

    // 等待新的 resync timestamp 或守護執行緒被終止
    std::unique_lock<std::mutex> lock(resync_mutex);
    resync_cond.wait(lock, [&] { return !is_resync_daemon_running || resync_next_timestamp != 0; });

    // 如果守護執行緒被終止，直接退出迴圈
    if (!is_resync_daemon_running) {
      lock.unlock();
      break;
    }

    // 將 目標觸發時間點 轉換為 system_clock::time_point
    auto t = std::chrono::system_clock::time_point { std::chrono::nanoseconds(resync_next_timestamp) };
    GError *err = NULL;

    // 關閉相機自動快門
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerMode", "On", &err);

    // 計算從現在到預期觸發時間的差距，加上100ms的buffer（確保未來時間）

    guint64 diff = (guint64)std::chrono::duration_cast<std::chrono::nanoseconds>
                   (std::chrono::system_clock::now() - t + std::chrono::milliseconds(100)).count();

    // choose dt
    guint64 dt = (this->resync_next_dt != 0) ? (guint64) this->resync_next_dt : this->trigger_nsec;

    // 根據 trigger_nsec(預期每幀間隔) 計算下一個對齊的觸發時間，
    // 並扣除 resync_offset(預期開啟快門跟收到第一幀的時間差距) 做微調
    auto trigger_off = t + std::chrono::nanoseconds( (guint64)(diff / dt * dt) - this->resync_offset);

    // 提前 50ms 醒來，此時快門應該已經關閉了
    std::this_thread::sleep_until(trigger_off - std::chrono::milliseconds(50));
    // 記錄觸發時間，用於後續 resync_offset 校正
    this->module_monitor->record_t = (guint64) std::chrono::duration_cast<std::chrono::nanoseconds>(trigger_off.time_since_epoch()).count();
    this->module_monitor->first_t = 0;

    // 等到真正要觸發關閉的時機
    std::this_thread::sleep_until(trigger_off);
    // 開啟相機自動快門，完成操作
    tcam_property_provider_set_tcam_enumeration(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerMode", "Off", &err);
    LOGMYSYSTEM << "resync at:"
                << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     trigger_off.time_since_epoch()).count()
                << " ns" << std::endl;

    // 稍後 100ms 後檢查是否有新的 first_t 時間戳，若有則更新 resync_offset
    std::this_thread::sleep_until(trigger_off + std::chrono::milliseconds(100));
    if (this->module_monitor->first_t) {
      this->resync_offset = this->module_monitor->first_t - this->module_monitor->record_t;
      LOGMYSYSTEM << "update resync_offset: " << this->resync_offset << " ns" << std::endl;
    }
    else {
      // 若沒收到有效的 first_t，表示模組未回報，視為錯誤
      //throw std::runtime_error("Error: No first_t timestamp.");
      LOGMYSYSTEM << "[Resync Daemon] Error: No first_t timestamp." << std::endl;
    }

    // 清除 resync timestamp 時間，進入等待狀態
    resync_next_timestamp = 0;

    lock.unlock();
  }
}

void TcamCamera::trigger_func() {
  GError* err = NULL;

  while (this->thread_running) {
    this->trigger_tp += std::chrono::nanoseconds(this->trigger_nsec);
    std::this_thread::sleep_until(this->trigger_tp);
    tcam_property_provider_set_tcam_command(TCAM_PROPERTY_PROVIDER(tcambin_), "TriggerSoftware", &err);
    if (err)
    {
      printf("!!! Could not trigger. !!!\n");
      printf("Error while setting trigger: %s\n", err->message);
      g_error_free(err);
      err = NULL;
    }
  }
}
