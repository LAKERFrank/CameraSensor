## init
- 需手動下載 weight，檔案放在 140.113.213.131 `/home/nol/bartek/camerasensor/LayerSensing/TrackNet/Tracknet1000/weights/best.pt`
- 請將該檔案複製到專案中的 `LayerSensing/TrackNet/Tracknet1000/weights` 路徑中，檔名不需調整

## 說明
- tracknet1000 主程式主要來是另外一個 git repo `https://github.com/BartekTao/ultralytics`，此專案是由 YOLOv8 官方專案 fork 出來的
- 目前採取的方式是人工複製檔案過來，若 inference 流程有更動，需人工同步兩邊的程式碼
- 有嘗試使用 submodule 的方式，但有出現 import 上的問題，沒有額外花時間去解決，未來若優化，建議可以將 tracknet1000 當作套件，publish 給大家使用