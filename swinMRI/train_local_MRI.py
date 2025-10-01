import os
import torch
import numpy as np
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split

# MONAI 相關匯入
from monai.data import (
    Dataset,
    ThreadDataLoader,
    decollate_batch,
)
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    SpatialPadd,
    RandRotate90d,
    Lambdad, # <--- 匯入 Lambdad
)
from torch.amp import GradScaler, autocast

# --- 1. 設定參數與環境 ---

# 資料路徑
DATA_DIR = "/mlsteam/data/spine/data/"

# 模型參數
IMG_SIZE = (96, 96, 96)
IN_CHANNELS = 1
# ## <--- 關鍵修改 1：設定正確的輸出通道數 ---
# 因為標籤將被對應到 0-20，所以總共有 21 個類別
OUT_CHANNELS = 21 
FEATURE_SIZE = 48

# 訓練參數
LEARNING_RATE = 1e-4
BATCH_SIZE = 24
NUM_EPOCHS = 100
VAL_INTERVAL = 5
NUM_WORKERS = 0
OUTPUT_DIR = "lab/saveModel"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. 檢查 AMD ROCm 設備 ---
if torch.cuda.is_available() and torch.version.hip:
    DEVICE = torch.device("cuda")
    print(f"ROCm 設備已找到！將使用: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("未找到 ROCm 設備，將使用 CPU 進行訓練。")

# --- 3. 資料準備 ---
def create_datalist_from_folders(data_dir: str):
    print(f"從 {data_dir} 掃描檔案...")
    data_dir = os.path.join(data_dir, '')
    image_paths = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    datalist = []
    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        corresponding_label = next((lbl_path for lbl_path in label_paths if os.path.basename(lbl_path) == img_filename), None)
        if corresponding_label:
            datalist.append({"image": img_path, "label": corresponding_label})
    return datalist

all_files = create_datalist_from_folders(DATA_DIR)
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
print(f"訓練集: {len(train_files)} 筆, 驗證集: {len(val_files)} 筆。")


# --- 4. 定義資料增強 (Transforms) ---

# ## <--- 關鍵修改 2：新增標籤重對應 (Label Remapping) 區塊 ---
print("="*50)
print(f"標籤重對應已啟用：將標籤 30 對應至 20。")
print("="*50)

# 使用查找表 (lookup table) 快速將 label==30 改成 20，其餘保持原值。
def remap_labels_func(label_tensor):
    """一個快速的重對應函數，將標籤 30 對應到 20"""
    device = label_tensor.device
    # 建立一個從 0 到 30 的身份對應查找表
    lookup_table = torch.arange(31, dtype=torch.long, device=device)
    # 將位置 30 的值（即原始標籤30）修改為新的目標值 20
    lookup_table[30] = 20
    # 應用查找表來轉換標籤
    return lookup_table[label_tensor.long()]
# ==============================================================

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader="NibabelReader", ensure_channel_first=True),
        # 在資料處理流程的一開始就進行標籤重對應
        Lambdad(keys="label", func=remap_labels_func),
        ScaleIntensityRanged(keys=["image"], a_min=-170, a_max=250, b_min=0.0, b_max=1.0, clip=True,),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.0), mode=("bilinear", "nearest"),),
        SpatialPadd(keys=["image", "label"], spatial_size=IMG_SIZE, method="symmetric"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=IMG_SIZE, pos=1, neg=1, num_samples=2, image_key="image", image_threshold=0,),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        EnsureTyped(keys=["image", "label"], device=DEVICE, track_meta=False),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader="NibabelReader", ensure_channel_first=True),
        # 驗證集也需要進行同樣的重對應
        Lambdad(keys="label", func=remap_labels_func),
        ScaleIntensityRanged(keys=["image"], a_min=-170, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=DEVICE, track_meta=False),
    ]
)


# --- 5. 建立 Dataset 和 DataLoader ---
# Dataset = 定義「資料怎麼準備」
# DataLoader = 定義「資料怎麼取出、怎麼送進模型」
print("建立資料集 (使用 monai.data.Dataset)...")
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

print("建立資料載入器 (使用 monai.data.ThreadDataLoader)...")
train_loader = ThreadDataLoader(train_ds, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
val_loader = ThreadDataLoader(val_ds, num_workers=NUM_WORKERS, batch_size=1)


# --- 6. 建立模型、損失函數和優化器 ---
print("正在建立模型...")
# 模型會自動使用更新後的 OUT_CHANNELS=21
model = SwinUNETR(img_size=IMG_SIZE, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, feature_size=FEATURE_SIZE, use_checkpoint=True,).to(DEVICE)
# 損失：Dice + CrossEntropy，設定 softmax + 將 y 轉 one-hot（多類分割）
# 損失函數會自動處理 21 個類別
loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
# 驗證指標會自動處理 21 個類別
post_label = AsDiscrete(to_onehot=OUT_CHANNELS)
post_pred = AsDiscrete(argmax=True, to_onehot=OUT_CHANNELS)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
best_metric = -1
best_metric_epoch = -1

# AMP：自動混合精度 + 梯度縮放，提升效能並避免 underflow
scaler = GradScaler(device=DEVICE.type)

# --- 7. 開始訓練迴圈 ---
print("="*20)
print("開始訓練")
print("="*20)
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
    
    for batch_data in progress_bar:
        images, labels = batch_data["image"], batch_data["label"]
        optimizer.zero_grad(set_to_none=True) 

        with autocast(device_type=DEVICE.type):
            outputs = model(images)
            loss = loss_function(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} 平均訓練損失: {avg_epoch_loss:.4f}")

    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"驗證中...", unit="case")
            for val_data in val_progress_bar:
                val_images, val_labels = val_data["image"], val_data["label"]
                # 以滑窗避免 3D 影像過大而 OOM
                with autocast(device_type=DEVICE.type):
                    val_outputs = sliding_window_inference(val_images, IMG_SIZE, 4, model)
                
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)

            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Epoch {epoch+1} 平均 Dice: {mean_dice:.4f}")
            
            if mean_dice > best_metric:
                best_metric = mean_dice
                best_metric_epoch = epoch + 1
                save_path = "best_metric_model.pth"
                torch.save(model.state_dict(), save_path)
                print(f"新紀錄！模型已儲存至 {save_path}")
            if (epoch + 1) % 20 == 0:
                periodic_save_path =  f"model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), periodic_save_path)
                print(f"已定期儲存備份模型至: {periodic_save_path}")
            # ---------------------------------

print("\n訓練完成！")
print(f"表現最好的模型在 Epoch {best_metric_epoch}，平均 Dice 為 {best_metric:.4f}")