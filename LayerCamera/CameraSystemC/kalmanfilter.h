#ifndef __KALMANFILTER_H__
#define __KALMANFILTER_H__

#include <Eigen/Dense>
#include <iostream>

class SimpleKalmanFilter {
public:
    SimpleKalmanFilter(double default_dt = 1.0 / 120.0);

    void step(double observed_timestamp);

    double timestamp() const {
        return x_(0);
    }

    double dt() const {
        return x_(1);
    }

private:
    Eigen::Vector2d x_;       // 狀態向量 [timestamp, dt]
    Eigen::Matrix2d P_;       // 誤差協方差
    Eigen::Matrix2d F_;       // 狀態轉移矩陣
    Eigen::Matrix2d Q_;       // 系統噪聲
    Eigen::Matrix2d R_;       // 觀測噪聲
    bool is_initialized_;
    double expected_dt_;
    double last_timestamp_;   // 上一次觀測值

    void predict();
    void update(Eigen::Vector2d z);
};

#endif // __KALMANFILTER_H__