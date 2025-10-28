import numpy as np
import matplotlib.pyplot as plt

class SimpleKalmanFilter:
    def __init__(self, initial_timestamp, initial_dt):
        # 狀態向量 [timestamp, dt]
        self.x = np.array([[initial_timestamp],
                           [initial_dt]])
        
        # 狀態轉移矩陣 F
        self.F = np.array([[1, 1],
                           [0, 1]])

        # 初始誤差協方差 P
        #self.P = np.eye(2) * 1.0
        self.P = np.array([[1.0, 1.0],
                           [1.0, 1.0]])
        
        # 系統噪聲 Q
        self.Q = np.array([[1e-8, 1e-8],
                           [1e-8, 1e-8]])  # 很小，表示系統自己很穩

        
        # 觀測噪聲 R
        self.R = np.array([[5e-4, 0.0],
                           [0.0, 5e-4]])   # 比較大，表示觀測值容易抖動

        # 前一次觀測的 timestamp
        self.last_timestamp = initial_timestamp

        self.ema_warmup_steps = 120*5
        self.ema_warmup_steps_remaining = self.ema_warmup_steps
        self.ema_dt = initial_dt  # 初始平滑值
        self.ema_max_alpha = 1e-2
        self.ema_min_alpha = 1e-5

    def isWarmup(self):
        return self.ema_warmup_steps_remaining > 0

    def step(self, observed_timestamp):
        """
        observed_timestamp: 新的時間戳記
        內部自動計算 dt
        """

        # 自動推算 dt
        observed_dt = observed_timestamp - self.last_timestamp
        self.last_timestamp = observed_timestamp

        # Predict
        self._predict()

        # Frame drop偵測
        expected_dt = self.x[1, 0]
        if observed_dt > 2 * expected_dt:
            # 用目前的dt值
            observed_dt = expected_dt  # Clamp到正常值
            # timestamp直接蓋掉
            self.x[0][0] = observed_timestamp
        else:
            if self.ema_warmup_steps_remaining:
                ratio = self.ema_warmup_steps_remaining / self.ema_warmup_steps
                self.ema_warmup_steps_remaining -= 1
            else:
                ratio = 0
            alpha = self.ema_max_alpha * (ratio) + self.ema_min_alpha * (1-ratio)
            # 使用 EMA 平滑 dt
            self.ema_dt = alpha * expected_dt + (1 - alpha) * self.ema_dt

        # 組合成完整的觀測向量 [timestamp, dt]
        z = np.array([[observed_timestamp],
                      [observed_dt]])

        # Update
        self._update(z)

    def _predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, z):
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K) @ self.P

    @property
    def timestamp(self):
        return self.x[0, 0]

    @property
    def dt(self):
        return self.ema_dt
        #return self.x[1, 0]


if __name__ == '__main__':
    data = [
        1745917295.412597,
        1745917295.420932,
        1745917295.429269,
        1745917295.437604,
        1745917295.445939,
        1745917295.454274,
        1745917295.462609,
        1745917295.470945,
        1745917295.479280,
        1745917295.487615,
        1745917295.495954,
        1745917295.504293,
        1745917295.512621,
        1745917295.520957,
        1745917295.529292,
        1745917295.537628,
        1745917295.545965,
        1745917295.554301,
        1745917295.562640,
        1745917295.570974,
        1745917295.579309,
        1745917295.587645,
        1745917295.595978,
        1745917295.604314,
        1745917295.612651,
        1745917295.620985,
        1745917295.629320,
        1745917295.637656,
        1745917295.645991,
        1745917295.654329,
        1745917295.662663,
        1745917295.670993,
        1745917295.679328,
        1745917295.687664,
        1745917295.696002,
        1745917295.704339,
        1745917295.712674,
        1745917295.721011,
        1745917295.729342,
        1745917295.737715,
        1745917295.746015,
        1745917295.754351,
        1745917295.762686,
        1745917295.771021,
        1745917295.779357,
        1745917295.787694,
        1745917295.796025,
        1745917295.804362,
        1745917295.812698,
        1745917295.821036,
        1745917295.829368,
        1745917295.837707,
        1745917295.846040,
        1745917295.854374,
        1745917295.862711,
        1745917295.871043,
        1745917295.879377,
        1745917295.887712,
        1745917295.896048,
        1745917295.904385,
        1745917295.912718,
        1745917295.921054,
        1745917295.929391,
        1745917295.937727,
        1745917295.946063,
        1745917295.954399,
        1745917295.962734,
        1745917295.971069,
        1745917295.979405,
        1745917295.987740,
        1745917295.996076,
        1745917296.004408,
        1745917296.012743,
        1745917296.021078,
        1745917296.029417,
        1745917296.037753,
        1745917296.046088,
        1745917296.054425,
        1745917296.062756,
        1745917296.071095,
        1745917296.079429,
        1745917296.087762,
        1745917296.096101,
        1745917296.104436,
        1745917296.112771,
        1745917296.121106,
        1745917296.129441,
        1745917296.137776,
        1745917296.146113,
        1745917296.154448,
        1745917296.162782,
        1745917296.171120,
        1745917296.179454,
        1745917296.187789,
        1745917296.196122,
        1745917296.204459,
        1745917296.212795,
        1745917296.221130,
    ]
    # 初始化 Kalman Filter
    dt_init = np.mean(np.diff(data))
    kf = SimpleKalmanFilter(data[0], dt_init)

    # 執行平滑
    filtered = []

    for measurement in data:
        filtered.append(kf.timestamp + kf.dt)
        kf.step(measurement)

    filtered = np.array(filtered)

    # 畫圖比較
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Original', marker='o')
    plt.plot(filtered, label='Kalman Filtered', marker='x')
    plt.legend()
    plt.xlabel('Frame Index')
    plt.ylabel('Timestamp (s)')
    plt.title('Timestamp Smoothing with Kalman Filter (Class Version)')
    plt.grid()
    plt.show()
