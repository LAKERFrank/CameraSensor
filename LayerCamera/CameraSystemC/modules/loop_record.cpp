#include "loop_record.h"

#define MAX_DURATION 10
#define FILE_MANAGE_FLAG 0  // 0: file size, 1: file count
#define MAX_TOTAL_SIZE (10 * 1024 * 1024 * 1024ULL)  // bytes
#define MAX_FILE_COUNT 5
#define FRAME_RATE 120
#define GOP_SIZE 15
#define BIT_RATE 2000
char loopRecordFolder[50] = "./video";

using namespace Module;

LoopRecord::LoopRecord(GstElement *mainPipeline, GstElement *mainTee,
                       unsigned int width, unsigned int height,
                       char loopRecordFolder1[50], int withTimestamp) {
  this->width_ = width;
  this->height_ = height;
  this->mainBin_ = mainPipeline;
  this->mainTee_ = mainTee;
  this->withTimestamp_ = withTimestamp;
  if (loopRecordFolder1) {
    strcpy(loopRecordFolder, loopRecordFolder1);
  }
  char createLoopRecordFolder[60] = "mkdir -p ";
  strcat(createLoopRecordFolder, loopRecordFolder);
  system(createLoopRecordFolder);
}

void LoopRecord::startLoopRecording() {
  GstElement *encoder_ = gst_element_factory_make("nvh264enc", "encoder");
  GstElement *parser_ = gst_element_factory_make("h264parse", "parser");
  GstElement *sink_ = gst_element_factory_make("splitmuxsink", "sink");
  GstElement *timeoverlay_ = NULL;
  GstElement *clockoverlay_ = NULL;
  queue_ = gst_element_factory_make("queue", "queue");
  GstElement *convert_ = gst_element_factory_make("videoconvert", "conv");

  assert(encoder_ && parser_ && sink_ && convert_);

  g_object_set(G_OBJECT(encoder_), "bitrate", (const char *)BIT_RATE, NULL);
  g_object_set(G_OBJECT(encoder_), "gop-size", GOP_SIZE, NULL);

  g_signal_connect(sink_, "format-location", G_CALLBACK(formatLocation), NULL);
  guint64 maxSizeTime =
      (guint64)((MAX_DURATION + (double)GOP_SIZE / (double)FRAME_RATE) * 1e9);
  g_object_set(sink_, "max-size-time", maxSizeTime, NULL);

  loopRecordBin_ = gst_bin_new("loopRecordBin_");
  gst_bin_add_many(GST_BIN(loopRecordBin_), queue_, convert_, encoder_, parser_,
                   sink_, NULL);

  if (!gst_element_link(queue_, convert_)) {
    std::cerr << "Failed to link queue_ to convert_!" << std::endl;
    gst_object_unref(mainBin_);
    return;
  }

  switch (withTimestamp_) {
    case 1:
      timeoverlay_ = gst_element_factory_make("timeoverlay", "timeoverlay");
      assert(timeoverlay_);
      gst_bin_add(GST_BIN(loopRecordBin_), timeoverlay_);
      if (!gst_element_link(convert_, timeoverlay_)) {
        std::cerr << "Failed to link convert_ to timeoverlay_!" << std::endl;
        gst_object_unref(mainBin_);
        return;
      }
      if (!gst_element_link(timeoverlay_, encoder_)) {
        std::cerr << "Failed to link timeoverlay_ to encoder_!" << std::endl;
        gst_object_unref(mainBin_);
        return;
      }
      break;

    case 2:
      clockoverlay_ = gst_element_factory_make("clockoverlay", "clockoverlay");
      assert(clockoverlay_);
      gst_bin_add(GST_BIN(loopRecordBin_), clockoverlay_);
      if (!gst_element_link(convert_, clockoverlay_)) {
        std::cerr << "Failed to link convert_ to clockoverlay_!" << std::endl;
        gst_object_unref(mainBin_);
        return;
      }
      if (!gst_element_link(clockoverlay_, encoder_)) {
        std::cerr << "Failed to link clockoverlay_to encoder_!" << std::endl;
        gst_object_unref(mainBin_);
        return;
      }
      break;

    default:
      if (!gst_element_link(convert_, encoder_)) {
        std::cerr << "Failed to link convert_ to encoder_!" << std::endl;
        gst_object_unref(mainBin_);
        return;
      }
      break;
  }

  if (!gst_element_link(encoder_, parser_)) {
    std::cerr << "Failed to link encoder_ to parser_!" << std::endl;
    gst_object_unref(mainBin_);
    return;
  }
  if (!gst_element_link(parser_, sink_)) {
    std::cerr << "Failed to link parser_ to sink_!" << std::endl;
    gst_object_unref(mainBin_);
    return;
  }

  gst_element_set_state(loopRecordBin_, GST_STATE_PLAYING);

  gst_bin_add(GST_BIN(mainBin_), loopRecordBin_);

  if (!gst_element_link(mainTee_, queue_)) {
    std::cerr << "Failed to link mainTee_ to queue_!" << std::endl;
    return;
  }

  gst_element_set_state(mainBin_, GST_STATE_PLAYING);

  isLoopRecording = true;
}

void LoopRecord::stopLoopRecording() {
  std::cerr << "stopLoopRecording started" << std::endl;
  isLoopRecordEos = false;

  gst_element_unlink(mainTee_, queue_);
  gst_bin_remove(GST_BIN(mainBin_), loopRecordBin_);
  // set main pipeline to PLAYING state
  gst_element_set_state(mainBin_, GST_STATE_PLAYING);

  // set record pipeline to PLAYING state
  gst_element_set_state(loopRecordBin_, GST_STATE_PLAYING);
  gst_element_get_state(loopRecordBin_, NULL, NULL, GST_CLOCK_TIME_NONE);

  // send EOS event
  gst_element_send_event(loopRecordBin_, gst_event_new_eos());

  std::unique_lock<std::mutex> lock(eos_mutex);
  eos_cond.wait(lock, [&] {
    isLoopRecordEos = true;
    std::cout << "sleep 1s for EOS" << std::endl;
    sleep(1);
    return isLoopRecordEos;
  });

  gst_element_set_state(loopRecordBin_, GST_STATE_NULL);

  reset();

  GstPad *sink_pad = gst_element_get_static_pad(queue_, "sink");
  GstPad *proxy_pad = gst_pad_get_peer(sink_pad);
  gst_pad_unlink(proxy_pad, sink_pad);
  gst_object_unref(proxy_pad);
  gst_object_unref(sink_pad);

  std::cerr << "stopLoopRecording completed" << std::endl;
}

void LoopRecord::reset() {
  isLoopRecording = false;
  isLoopRecordEos = false;
}

// File count and deletion
// Remove files if the total size exceeds the maximum limit
void LoopRecord::manageFileSize(const char *directory) {
  GDir *dir = g_dir_open(directory, 0, NULL);
  if (!dir) return;

  GList *files = NULL;
  const gchar *fileName;

  gsize totalSize = 0;
  static long long maxSizePerFile = 0;

  while ((fileName = g_dir_read_name(dir))) {
    gchar *filePath = g_build_filename(directory, fileName, NULL);
    files = g_list_append(files, filePath);

    struct stat st;
    if (stat(filePath, &st) == 0) {
      totalSize += st.st_size;
      if (st.st_size > maxSizePerFile) {
        maxSizePerFile = st.st_size;
      }
    }
  }
  g_dir_close(dir);

  std::cout << "Size: (" << totalSize << "/" << MAX_TOTAL_SIZE << ")"
            << std::endl;

  if (totalSize + maxSizePerFile > MAX_TOTAL_SIZE) {
    files = g_list_sort(files, (GCompareFunc)g_strcmp0);
    remove((char *)g_list_nth_data(files, 0));
    std::cout << "remove: " << (char *)g_list_nth_data(files, 0) << std::endl;
  }

  g_list_free_full(files, g_free);
}

// File count and deletion
// Remove files if the count exceeds the maximum limit
void LoopRecord::manageFileCount(const char *directory) {
  GDir *dir = g_dir_open(directory, 0, NULL);
  if (!dir) return;

  GList *files = NULL;
  const gchar *fileName;

  while ((fileName = g_dir_read_name(dir))) {
    gchar *filePath = g_build_filename(directory, fileName, NULL);
    files = g_list_append(files, filePath);
  }
  g_dir_close(dir);

  // Print the current file count
  std::cout << "File count: " << g_list_length(files) << "/" << MAX_FILE_COUNT
            << std::endl;

  // Remove files if the count exceeds the maximum limit
  if (g_list_length(files) + 1 > MAX_FILE_COUNT) {
    files = g_list_sort(files, (GCompareFunc)g_strcmp0);  // Sort by filename
    gchar *oldestFile = (gchar *)g_list_nth_data(files, 0);
    if (oldestFile) {
      remove(oldestFile);
      std::cout << "Removed: " << oldestFile << std::endl;
    }
  }

  g_list_free_full(files, g_free);
}

// Dynamically retrieve recording time
gchar *LoopRecord::formatLocation(GstElement *splitmux, guint fragmentId,
                                  gpointer userData) {
  switch (FILE_MANAGE_FLAG) {
    case 0:
      manageFileSize(loopRecordFolder);
      break;
    case 1:
      manageFileCount(loopRecordFolder);
      break;
  }

  GDateTime *now = g_date_time_new_now_local();
  gchar *timestamp = g_date_time_format(now, "%Y%m%d_%H:%M:%S.%f");
  g_date_time_unref(now);

  gchar milliseconds[4];
  g_strlcpy(milliseconds, timestamp + strlen(timestamp) - 6, 4);

  gchar *finalTimestamp = g_strdup_printf(
      "%.*s%s", (int)(strlen(timestamp) - 6), timestamp, milliseconds);

  g_free(timestamp);

  return g_strconcat(loopRecordFolder, "/", finalTimestamp, ".mp4", NULL);
}