# YOLOv5 手勢辨識系統 

本專案採用 YOLOv5 架構，針對 HAGRID (Hand Gesture Recognition Image Dataset) 資料集進行自定義模型訓練與推論分析。專案目標為建構一套具備即時辨識能力的手勢判定流程，並進行推論延遲 (Inference Latency) 與精準度 (mAP) 的效能評估。

## 📁 專案架構 

本儲存庫包含模型訓練運行紀錄、推論分析腳本及執行環境設定：

* **`yolov5_runtime/`**: YOLOv5 核心執行檔與模型依賴套件。
* **`datasets/HAGRID-YOLO/`**: 訓練與驗證所使用之 HAGRID 資料集存放路徑。本專案共取用 7,600 張影像（Train: 6,000, Val: 800, Test: 800），涵蓋 like, nogesture, ok, peace, stop 五種類別。
* **`artifacts/runs/train/`**: 模型訓練過程的權重檔 (Weights) 與指標紀錄 (Metrics)。
* **`demo/`**: Live Demo 與測試腳本。
* **`gesture_inference_analysis.ipynb`**: 透過 Colab 進行模型推論與效能分析之核心筆記本。
* **`colab_env.toml` & `colab_check_paths.py`**: Colab 雲端環境之依賴設定與路徑驗證腳本。

## ⚙️ 開發環境與參數設定 

* **模型架構**: YOLOv5
* **硬體環境**: NVIDIA GPU (CUDA)
* **訓練參數** (最佳模型配置):
    * Batch Size: 32
    * Learning Rate: 0.01
    * Epochs: 30

## 📊 效能指標 

以下數據為最佳模型在驗證集上的客觀表現：

* **mAP@0.5**: 0.990
* **mAP@0.5:0.95**: 0.803
* **推論延遲 (Inference Latency)**: 純模型推論平均需時 25.253 ms，包含影像讀取與前後處理之總延遲為 29.434 ms，符合即時影像串流之辨識需求。

## 🚀 執行說明 

**1. 環境建置**
請確保本地端或 Colab 環境已安裝相關依賴套件。若使用 Colab，請先執行環境設定檔：
```bash
python colab_check_paths.py
