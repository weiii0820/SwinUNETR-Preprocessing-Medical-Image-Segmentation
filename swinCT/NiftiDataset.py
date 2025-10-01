import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    Resized,
    EnsureTyped,
)

class NiftiDataset(Dataset):
    """
    用於讀取 NIfTI 格式的 3D 醫學影像和標籤的資料集。
    """
    def __init__(self, image_dir, label_dir, target_size):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size

        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(('.nii', '.nii.gz'))])
        
        assert len(self.image_files) == len(self.label_files), "影像和標籤的檔案數量不匹配！"

        self.transforms = Compose([
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            # 影像 intensity normalize（與 predict.py 相同範圍）
            ScaleIntensityRanged(
                keys=["image"], a_min=-170, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            # 以影像前景裁切，image/label 同步裁切
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # 對齊到 RAS
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # 統一 voxel spacing；影像用雙線性、標籤用最近鄰
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
            Spacingd(keys=["label"], pixdim=(1.5, 1.5, 1.0), mode="nearest"),
            
            # 整圖（補零/裁切，不拉伸）
            #ResizeWithPadOrCropd(keys=["image","label"], spatial_size=self.target_size),

            # ✅ 整體縮放到固定張量大小（避免裁切/補零）
            # 影像使用三線性；標籤使用最近鄰以避免類別混疊
            # Resized(
            #     keys=["image", "label"],
            #     spatial_size=self.target_size,
            #     mode=("trilinear", "nearest"),
            #     align_corners=(False, None),
            # ),

            # 放到當前裝置/型別
            EnsureTyped(keys=["image", "label"]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        data_dict = {"image": self.image_files[idx], "label": self.label_files[idx]}
        processed_data = self.transforms(data_dict)
    
        # --- 修正部分 ---
        # 直接回傳處理後的字典。
        # processed_data['image'] 此時是一個包含元數據的 MetaTensor 物件。
        return processed_data