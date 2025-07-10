// mainwindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <gst/gst.h>
#include <gst/video/videooverlay.h>

#include <QGridLayout>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QTime>
#include <QTimer>

// #include "tcamcamera.h"
#include "recorder.h"
// #include <QWidget>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class MainWindow : public QMainWindow {
  // Q_OBJECT

 public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

 protected:
 private slots:
  void onClicked();
  void updateTime();

 private:
  //   void opencam(int idx, std::string serial, int width, int height, int
  //   fps);
  // start all cameras
  void start_recording();
  // stop all cameras
  void stop_recording();

  void startLoopRecording();

  void stopLoopRecording();

  void takeSnapshot();
  // start single camera
  //   void start_recording_cam(int idx, fs::path recorddir);
  //   // stop single camera
  //   void stop_recording_cam(int idx);

 private:
  QWidget *window;
  QGridLayout *mainLayout;
  //   std::vector<std::shared_ptr<gsttcam::TcamCamera>> cams;
  //   // std::vector<gsttcam::TcamCamera *> cams;
  //   std::vector<gsttcam::CameraInfo> camDevices;
  std::vector<QWidget *> videoWidgets;
  QPushButton *button;
  QPushButton *button_snapshot;
  QPushButton *buttonLoopRecord_;
  QLabel *timeLabel;
  QTimer *timer;
  QTime startTime;
  //   int elapsedSeconds = 0;
  int elapsedMilliseconds = 0;
  //   bool isStreaming;
  bool isRecording;
  bool isLoopRecording;
  fs::path ROOTDIR;
  //   std::map<std::string, std::map<std::string, std::string>> mainconfig;
  std::shared_ptr<Recorder> recorder;
  std::vector<unsigned long> winIds;
  // Recorder recorder;
};

#endif  // MAINWINDOW_H
