#include "mainwindow.h"

#define NODE_NUM 1

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  // std::cout << "main pid = " << getpid() << "\n";
  // std::cout << "main tid = " << gettid() << "\n";

  try {
    fs::path currentPath = fs::current_path();
    ROOTDIR = currentPath.parent_path().parent_path();
    recorder = std::make_shared<Recorder>(ROOTDIR.string());
  } catch (const fs::filesystem_error &err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return;
  }

  window = new QWidget(this);
  setCentralWidget(window);
  mainLayout = new QGridLayout;
  window->setLayout(mainLayout);

  videoWidgets.resize(NODE_NUM);
  winIds.resize(NODE_NUM);

  for (int i = 0; i < NODE_NUM; i++) {
    videoWidgets[i] = new QWidget(this);
    videoWidgets[i]->setFixedWidth(512);
    videoWidgets[i]->setFixedHeight(384);
    // videoWidget->setAttribute(Qt::WA_PaintOnScreen);
    // videoWidget->setAttribute(Qt::WA_OpaquePaintEvent);
    // videoWidgets.push_back(videoWidget);
    // mainLayout->addWidget(videoWidgets[i], i / 2 + 1, i % 2);
    winIds[i] = videoWidgets[i]->winId();
  }

  mainLayout->addWidget(videoWidgets[0], 1, 2);
  // mainLayout->addWidget(videoWidgets[1], 1, 0);
  // mainLayout->addWidget(videoWidgets[2], 3, 0);
  // mainLayout->addWidget(videoWidgets[3], 3, 2);
  // mainLayout->addWidget(videoWidgets[4], 1, 1);
  // mainLayout->addWidget(videoWidgets[5], 3, 1);

  recorder->init("44224010", winIds[0], "127.0.0.1", 9000, 0);

  button = new QPushButton("start");
  button->setFixedHeight(50);
  // button->setFixedWidth(200);
  mainLayout->addWidget(button, 0, 0);
  connect(button, &QPushButton::clicked, this, [&]() {
    if (isRecording) {
      stop_recording();
    } else {
      start_recording();
    }
  });

  button_snapshot = new QPushButton("snapshot");
  button_snapshot->setFixedHeight(50);
  // button->setFixedWidth(200);
  mainLayout->addWidget(button_snapshot, 0, 2);
  connect(button_snapshot, &QPushButton::clicked, this,
          [&]() { takeSnapshot(); });

  buttonLoopRecord_ = new QPushButton("startLoopRecord");
  buttonLoopRecord_->setFixedHeight(50);
  mainLayout->addWidget(buttonLoopRecord_, 1, 0);
  connect(buttonLoopRecord_, &QPushButton::clicked, this, [&]() {
    if (isLoopRecording) {
      stopLoopRecording();
    } else {
      startLoopRecording();
    }
  });

  timer = new QTimer(this);
  timer->setInterval(
      100);  // Set interval to 100 milliseconds for 0.1-second precision
  connect(timer, &QTimer::timeout, this, &MainWindow::updateTime);

  timeLabel = new QLabel(this);
  timeLabel->setText("00:00:00.0");
  timeLabel->setAlignment(Qt::AlignCenter);
  timeLabel->setStyleSheet("font-size: 28px;");
  mainLayout->addWidget(timeLabel, 0, 1);
  isRecording = false;
  isLoopRecording = false;
}

MainWindow::~MainWindow() { return; }

void MainWindow::updateTime() {
  QTime currentTime = QTime::currentTime();
  elapsedMilliseconds = startTime.msecsTo(currentTime);
  QTime displayTime = QTime(0, 0, 0).addMSecs(elapsedMilliseconds);

  timeLabel->setText(displayTime.toString("hh:mm:ss.z"));
}

void MainWindow::takeSnapshot() {
  time_t curr_time;
  tm *curr_tm;
  char datetime_string[100];
  time(&curr_time);
  curr_tm = localtime(&curr_time);
  fs::create_directory("snapshots");

  strftime(datetime_string, 50, "%Y%m%d_%H%M%S", curr_tm);

  std::shared_ptr<gsttcam::Snapshot> s = recorder->takeSnapshot();

  // save image file

  char filename[256];
  sprintf(filename, "snapshots/image_%s_cam%d.jpg", datetime_string, 0);
  cv::imwrite(filename, *s->toCvMat().get());

  LOGMYSYSTEM << "saved image file " << filename << std::endl;
}

void MainWindow::start_recording() {
  if (isRecording) {
    return;
  }
  recorder->startRecording("hello_test");
  button->setText("stop");
  timer->start();
  startTime = QTime::currentTime();
  isRecording = true;
}

void MainWindow::stop_recording() {
  if (!isRecording) {
    return;
  }
  recorder->stopRecording();
  button->setText("start");
  timer->stop();
  isRecording = false;
}

void MainWindow::startLoopRecording() {
  if (isLoopRecording) {
    return;
  }
  recorder->startLoopRecording();
  buttonLoopRecord_->setText("stopLoopRecord");
  timer->start();
  startTime = QTime::currentTime();
  isLoopRecording = true;
}

void MainWindow::stopLoopRecording() {
  if (!isLoopRecording) {
    return;
  }
  recorder->stopLoopRecording();
  buttonLoopRecord_->setText("startLoopRecord");
  timer->stop();
  isLoopRecording = false;
}