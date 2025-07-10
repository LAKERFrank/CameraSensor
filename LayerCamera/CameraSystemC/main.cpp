// main.cpp
#include <QApplication>
#include <gst/gst.h>

#include "mainwindow.h"

int main(int argc, char *argv[]) {
  setlinebuf(stdout);
  gst_init(&argc, &argv);
  QApplication app(argc, argv);
  MainWindow mainWindow;
  //   mainWindow.showMaximized();
  mainWindow.resize(1400, 1000);
  mainWindow.show();
  return app.exec();
}
