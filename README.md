# 醫學影像分割中的前處理策略影響  
*以 SwinUNETR 模型在 MRI 與 CT 資料集上的應用為例*

## 📌 專案簡介
本專題研究 **不同前處理策略** 對醫學影像分割效能的影響。  
我們採用 **SwinUNETR**（結合 Swin Transformer 與 U-Net 的 3D 分割架構），並在 **MRI** 與 **CT** 兩種模態上進行比較。  

研究重點：  
- 🔹 **推論方式**：Sliding-window 與 整圖裁切的比較  
- 🔹 **評估方式**：Dice 指標是否包含背景類別  

結果顯示，**前處理方式與評估設定會顯著影響分割效能**，在臨床應用中應建立一致的標準。  

---

## 📊 資料集
- **MRI**：Lumbar Spine MRI Dataset（48,345 張切片，21 個標註類別）  
- **CT**：Spine CT Segmentation Dataset（1,089 個掃描影像，26 個標註類別）  

兩個資料集皆為專家人工標註，並分為訓練集、驗證集與測試集。  

---

## 🏗️ 研究方法
- **模型**：SwinUNETR (Transformer + U-Net 架構)  
- **前處理**：  
  - 重採樣至 (1.5, 1.5, 1.0) mm  
  - 強度正規化（CT：-170~250 HU，MRI：[0,1]）  
  - 前景裁切  
  - 數據增強（翻轉、旋轉、強度偏移）  

- **損失函數**：DiceCELoss = Dice Loss + Cross Entropy  
- **最佳化器**：AdamW（lr=1e-4, weight decay=1e-5）  
- **訓練**：100 epochs，啟用 AMP（自動混合精度）  

---

## 📈 實驗結果
### 1. 推論方式比較
- Sliding-window 明顯優於整圖裁切  
- 在 CT 分割中特別顯著（p < 0.001）  

### 2. 評估方式比較
- 不含背景的 Dice 分數，更能反映臨床應用需求，尤其在 CT 上差異顯著（p = 0.050）  
- MRI 在兩種設定下皆表現穩定  

---

## 📂 專案結構
```
├── Report/ # 專題報告 PDF
├── swinCT/ # CT 模型程式碼
│ ├── model.py
│ ├── train_local_CT.py
│ ├── NiftiDataset.py
│ └── test.py
├── swinMRI/ # MRI 模型程式碼
│ ├── model.py
│ ├── train_local_MRI.py
│ ├── t_NiftiDataset.py
│ └── t_testwithoutoutput.py
└── README.md
```

---

## 📥 資料及來源
由於 GitHub 檔案大小限制，完整測試資料與訓練好的模型參數請至外部下載：  

- [MRI 資料集（Mendeley）](https://data.mendeley.com/datasets/zbf6b4pttk/2)  
- [CT 資料集（Kaggle）](https://www.kaggle.com/datasets/pycadmk/spine-segmentation-from-ct-scans)  
