#ifndef MODULE_LOOP_RECORD_H__
#define MODULE_LOOP_RECORD_H__

#include <glib.h>
#include <gst/gst.h>
#include <sys/stat.h>

#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>

namespace Module {
class LoopRecord {
 public:
  LoopRecord(GstElement *mainPipeline, GstElement *mainTee, unsigned int width,
             unsigned int height, char loopRecordFolder1[50],
             int withTimestamp);
  ~LoopRecord() {};

  void startLoopRecording();
  void stopLoopRecording();
  void reset();

 private:
  static gchar *formatLocation(GstElement *splitmux, guint fragmentId,
                               gpointer userData);
  static void manageFileSize(const char *directory);
  static void manageFileCount(const char *directory);

  GstElement *mainBin_ = nullptr;
  GstElement *mainTee_ = nullptr;

  GstElement *loopRecordBin_ = nullptr;
  GstElement *queue_ = nullptr;

  bool isLoopRecordEos = false;
  bool isLoopRecording = false;
  std::condition_variable eos_cond;
  std::mutex eos_mutex;

  // not modified under reset
  unsigned int width_;
  unsigned int height_;
  int withTimestamp_;
};
};  // namespace Module

#endif
