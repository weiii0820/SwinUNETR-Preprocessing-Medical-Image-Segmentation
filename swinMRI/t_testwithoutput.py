# t_test.py  —— 儲存流程改為 predict.py 風格的 Invertd + SaveImaged
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import numpy as np

from monai.transforms import (
    Compose, Activations, AsDiscrete,
    EnsureTyped, Invertd, SaveImaged, Lambdad
)
from monai.metrics import DiceMetric
from monai.utils import MetricReduction
from monai.inferers import sliding_window_inference
from monai.data import MetaTensor

# 自訂
from t_NiftiDataset import NiftiDataset
from model import get_swin_unetr_model

# ========= 參數 =========
MODEL_PATH = "./best_metric_model0709.pth"
TEST_CT_DIR = "./test_image"
TEST_LABEL_DIR = "./test_label"
OUTPUT_DIR = "./t_testwithoutput"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {DEVICE}")

# Swin-UNETR 建置參數（依你的訓練設定）
IN_CHANNELS = 1
OUT_CHANNELS = 21
FEATURE_SIZE = 48
ROI_SIZE = (128, 128, 128)
SW_BATCH = 1 
OVERLAP = 0.25 # 相鄰 patch 的重疊比例

# ========= 準備模型 =========
print(f"正在從 {MODEL_PATH} 載入模型…")
model = get_swin_unetr_model(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    feature_size=FEATURE_SIZE
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("模型載入完成。")

# ========= 準備資料 =========
print("正在準備測試資料…")
# ※ 這個 Dataset 需與 predict.py 使用相同的前處理（invertible transforms）
test_dataset = NiftiDataset(TEST_CT_DIR, TEST_LABEL_DIR, ROI_SIZE)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# 取得與 Dataset 相同的前處理 transform（供 Invertd 反轉）
_pre_xforms = test_dataset.transforms

# ========= 後處理 & 儲存（predict.py 風格） =========

# 1) 給 Dice：logits -> softmax -> argmax -> one-hot(21)
post_pred_for_dice = Compose([
    Activations(softmax=True, dim=1),          # (B,C,D,H,W)
    AsDiscrete(argmax=True, arg_dim=1, to_onehot=OUT_CHANNELS, dim=1),
])
post_label_for_dice = Compose([
    AsDiscrete(to_onehot=OUT_CHANNELS, dim=1),
])

# 2) 給存檔：沿用 predict.py 思路，在 post_transforms 內完成：
#    logits(C,D,H,W) -> softmax -> argmax(1ch) -> 20→30 -> Invertd -> SaveImaged
def _inverse_remap_func(pred_tensor):
    # 將 20 改回 30（僅用於輸出檔）
    remapped = pred_tensor.clone()
    remapped[pred_tensor == 20] = 30
    return remapped

post_transforms_for_save = Compose([
    EnsureTyped(keys="pred"),                                       # 確保 tensor/meta 型別
    Lambdad(keys="pred", func=lambda x: torch.softmax(x, dim=0)),   # C-first (無 batch) 的 softmax
    Lambdad(keys="pred", func=lambda x: torch.argmax(x, dim=0, keepdim=True)),  # -> (1,D,H,W)
    Lambdad(keys="pred", func=_inverse_remap_func),
    Invertd(
        keys="pred",
        transform=_pre_xforms if _pre_xforms is not None else None,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        nearest_interp=True,
        to_tensor=True,
    ),
    SaveImaged(
        keys="pred",
        meta_keys="pred_meta_dict",
        output_dir=OUTPUT_DIR,
        output_postfix="_seg",
        resample=False,
        separate_folder=False,
    ),
])

dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)

# ========= 推論 & 儲存 =========
print("開始進行推論 (Inference)…")
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, desc="Testing")):
        # Dataset 需提供 MONAI 的 MetaTensor（常見於 LoadImaged + EnsureTyped 後）
        image = data["image"].to(DEVICE)   # (B,1,D,H,W)
        label = data["label"].to(DEVICE)   # (B,1,D,H,W)

        # 滑窗推論
        logits = sliding_window_inference(
            inputs=image,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH,
            predictor=model,
            overlap=OVERLAP,
        )  # (B,C,D,H,W)

        # 整圖縮放
        #logits = model(image)

        # ====== Dice 計算（保持你原本流程）======
        pred_oh = post_pred_for_dice(logits)      # (B,21,D,H,W)
        label_oh = post_label_for_dice(label)     # (B,21,D,H,W)
        print("pred_onehot:", tuple(pred_oh.shape), "label_onehot:", tuple(label_oh.shape))
        dice_metric(y_pred=pred_oh, y=label_oh)

        # ====== 儲存（predict.py 風格）======
        # 取出 batch 維度（B=1），把 logits 轉為 MetaTensor，沿用來源影像的 meta
        # 注意：post_transforms_for_save 會在內部做 softmax/argmax/20→30/反變換/儲存
        # 為了 Invertd，需要同時把 "image" 放進 dict 裡提供原始的 meta stack
        image_mt: MetaTensor = image[0]  # (1,D,H,W) 的 MetaTensor
        pred_mt  = MetaTensor(logits[0], meta=image_mt.meta.copy())  # (C,D,H,W)

        # 組裝 dict，讓 Invertd 參考到 image 的 invertible transforms
        out_dict = {
            "image": image_mt,   # 原影像（含 invertible transform 的紀錄）
            "pred":  pred_mt,    # 模型輸出（C,D,H,W）
            # Invertd/SaveImaged 會使用 pred_meta_dict；這裡讓它從 pred 的 meta 拷貝
            "pred_meta_dict": image_mt.meta.copy()
        }

        # 執行儲存流程（與 predict.py 一致）
        try:
            post_transforms_for_save(out_dict)
            # SaveImaged 會自動依原檔名在 OUTPUT_DIR 產生 {stem}_seg.nii.gz
            # 若你要閱讀實際輸出檔名，可用 out_dict["pred_meta_dict"]["filename_or_obj"]
        except Exception as e:
            # 萬一 Dataset 未提供可反變換的 transform，也能給出明確訊息
            print(f"[警告] 檔案儲存（Invertd/SaveImaged）失敗：{e}")
            # 退而求其次：至少把反變換前的結果存一份（避免完全沒輸出）
            # （保底作法：與原本 nib.save 類似，但不做空間反變換）
            stem = Path(str(image_mt.meta.get("filename_or_obj", f"test_case_{i:04d}"))).name
            if stem.endswith(".nii.gz"):
                stem = stem[:-7]
            else:
                stem = Path(stem).stem
            fallback_path = Path(OUTPUT_DIR) / f"{stem}_seg_rawspace.nii.gz"

            # 這裡沿用 post_transforms_for_save 的前半：softmax/argmax/20→30
            pred_np = torch.softmax(logits[0], dim=0).argmax(dim=0, keepdim=True)
            pred_np[pred_np == 20] = 30
            pred_np = pred_np[0].cpu().numpy().astype(np.uint8)

            import nibabel as nib
            affine = image_mt.meta.get("original_affine", None) or image_mt.meta.get("affine", None)
            if affine is None:
                affine = np.eye(4, dtype=float)
            nib.save(nib.Nifti1Image(pred_np, affine), str(fallback_path))
            print(f"✔ 已輸出（不含反變換保底檔）：{fallback_path.name}")

mean_dice = dice_metric.aggregate().item()
print("\n測試完成！")
print(f"所有測試資料的平均 Dice Score：{mean_dice:.4f}")
print(f"預測結果已儲存至：{OUTPUT_DIR}（或保底檔 *_seg_rawspace.nii.gz）")