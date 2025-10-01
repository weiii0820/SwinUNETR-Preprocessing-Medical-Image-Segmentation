import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader
import os
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import numpy as np

# 從我們自己寫的檔案中匯入
from NiftiDataset import NiftiDataset
from model import get_swin_unetr_model
from monai.transforms import (
    Compose, Activations, AsDiscrete,
    EnsureTyped, Invertd, SaveImaged, Lambdad,
    KeepLargestConnectedComponent, FillHoles
)
from monai.metrics import DiceMetric
from monai.utils import MetricReduction
from monai.inferers import sliding_window_inference
from monai.data import MetaTensor

# --- 1. 設定參數 ---
MODEL_PATH = "./best_metric_model0622.pth"
TEST_CT_DIR = "./test_image"
TEST_LABEL_DIR = "./test_label"
OUTPUT_DIR = "./test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (128, 128, 128)
IN_CHANNELS = 1
OUT_CHANNELS =26
FEATURE_SIZE = 48
SW_BATCH = 1 
OVERLAP = 0.25

# --- 2. 準備模型和資料 ---
print("正在設定設備...")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"CUDA 設備已找到！將使用: {torch.cuda.get_device_name(0)}")
elif torch.version.hip:
    DEVICE = torch.device("cuda")
    print("ROCm 設備已找到！")
else:
    DEVICE = torch.device("cpu")
    print("未找到 GPU 設備，將使用 CPU 進行推論。")

print(f"正在從 {MODEL_PATH} 載入模型...")
model = get_swin_unetr_model(img_size=IMG_SIZE, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, feature_size=FEATURE_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("模型載入完成。")

print("正在準備測試資料...")
test_dataset = NiftiDataset(image_dir=TEST_CT_DIR, label_dir=TEST_LABEL_DIR, target_size=IMG_SIZE)
# --- 修正點 1: 將 num_workers 設為 0 ---
# 這有助於在所有環境下都穩定地獲取元數據
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# 取得與 Dataset 相同的前處理 transform（供 Invertd 反轉）
_pre_xforms = test_dataset.transforms

# ========= 後處理 & 儲存 =========
# 1) 給 Dice：logits -> softmax -> argmax -> one-hot(26)
post_pred_for_dice = Compose([
    Activations(softmax=True, dim=1),          # (B,C,D,H,W)
    AsDiscrete(argmax=True, arg_dim=1, to_onehot=OUT_CHANNELS, dim=1),
])
post_label_for_dice = Compose([
    AsDiscrete(to_onehot=OUT_CHANNELS, dim=1),
])

post_transforms_for_save = Compose([
    EnsureTyped(keys="pred"),                                       # 確保 tensor/meta 型別
    Lambdad(keys="pred", func=lambda x: torch.softmax(x, dim=0)),   # C-first (無 batch) 的 softmax
    Lambdad(keys="pred", func=lambda x: torch.argmax(x, dim=0, keepdim=True)),  # -> (1,D,H,W)
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
        resample=True,
        separate_folder=False,
    ),
])

dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)

# --- 3. 開始測試迴圈 ---
print("開始進行推論 (Inference)...")
with torch.no_grad():
    # --- 修正點 2: 使用 enumerate 來獲取索引 i ---
    for i, test_data in enumerate(tqdm(test_loader, desc="Testing")):
        image = test_data["image"].to(DEVICE)
        label = test_data["label"].to(DEVICE)

        # 滑窗推論
        logits = sliding_window_inference(
            inputs=image,
            roi_size=IMG_SIZE,
            sw_batch_size=SW_BATCH,
            predictor=model,
            overlap=OVERLAP,
        )  # (B,C,D,H,W)

        # 整圖縮放
        # logits = model(image)

        # ====== Raw Dice ======
        pred_oh = post_pred_for_dice(logits)      # (B,26,D,H,W)
        label_oh = post_label_for_dice(label)     # (B,26,D,H,W)
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

            # 這裡沿用 post_transforms_for_save 的前半：softmax/argmax
            pred_np = torch.softmax(logits[0], dim=0).argmax(dim=0, keepdim=True)
            pred_np = pred_np[0].cpu().numpy().astype(np.uint8)

            import nibabel as nib
            affine = image_mt.meta.get("original_affine", None) or image_mt.meta.get("affine", None)
            if affine is None:
                affine = np.eye(4, dtype=float)
            nib.save(nib.Nifti1Image(pred_np, affine), str(fallback_path))
            print(f"✔ 已輸出（不含反變換保底檔）：{fallback_path.name}")

mean_dice = dice_metric.aggregate().item()
print(f"\n測試完成！")
print(f"所有測試資料的平均 Dice Score：{mean_dice:.4f}")
print(f"預測結果已儲存至資料夾: {OUTPUT_DIR}")