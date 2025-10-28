#include "kalmanfilter.h"

SimpleKalmanFilter::SimpleKalmanFilter(double default_dt)
    : is_initialized_(false), expected_dt_(default_dt) 
{
    x_.setZero();  // 還沒用到，第一次 step 再設

    // 狀態轉移矩陣 F
    F_ << 1, 1,
          0, 1;

    // 初始誤差協方差 P
    P_ << 1, 1,
          1, 1;

    // 系統噪聲 Q
    Q_ << 1e-8, 1e-8,
          1e-8, 1e-8;

    // 觀測噪聲 R
    R_ << 5e-4,    0,
             0, 5e-4;
};

void SimpleKalmanFilter::step(double observed_timestamp) {
    if (!is_initialized_) {
        // 第一次呼叫，初始化狀態
        x_ << observed_timestamp, expected_dt_;
        last_timestamp_ = observed_timestamp;
        is_initialized_ = true;
        return;
    }

    // 推算 dt
    double observed_dt = observed_timestamp - last_timestamp_;
    last_timestamp_ = observed_timestamp;

    this->predict();

    double estimated_dt = x_(1); // 來自Kalman的估計
    // Frame drop 偵測與處理
    if (observed_dt > 2 * estimated_dt) {
        observed_dt = estimated_dt;
        // 回推 timestamp
        x_(0) = observed_timestamp;
    }

    // 組成觀測量 z
    Eigen::Vector2d z;
    z << observed_timestamp, observed_dt;

    this->update(z);
}

void SimpleKalmanFilter::predict() {
    // Predict
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void SimpleKalmanFilter::update(Eigen::Vector2d z) {
    // Update
    Eigen::Vector2d y = z - x_;
    Eigen::Matrix2d S = P_ + R_;
    Eigen::Matrix2d K = P_ * S.inverse();

    x_ = x_ + K * y;
    P_ = (Eigen::Matrix2d::Identity() - K) * P_;
}
