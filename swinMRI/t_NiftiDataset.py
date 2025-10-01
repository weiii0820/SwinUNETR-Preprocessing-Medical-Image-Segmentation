# NiftiDataset.py
import os
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
    Lambdad,
)

class NiftiDataset(Dataset):
    """
    讀取 NIfTI 3D 影像與標籤，前處理流程改為與 predict.py 的 pre_transforms 一致：
    Load → EnsureChannelFirst → ScaleIntensityRange(影像) → CropForeground(依影像) →
    Orientation(RAS) → Spacing(1.5,1.5,1.0) → EnsureTyped
    並保留 label 30→20 的重編碼，讓訓練/評估使用 0..20 類別空間。
    """
    def __init__(self, image_dir, label_dir, target_size):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size

        self.image_files = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir)
             if f.endswith((".nii", ".nii.gz"))]
        )
        self.label_files = sorted(
            [os.path.join(label_dir, f) for f in os.listdir(label_dir)
             if f.endswith((".nii", ".nii.gz"))]
        )
        assert len(self.image_files) == len(self.label_files), "影像與標籤檔案數量不一致！"

        # 和 predict.py 一致的影像前處理；label 做對應幾何轉換（nearest）
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
            
            # 標籤重編碼：把 30 映到 20（模型/評估使用 0..20）
            Lambdad(keys="label", func=lambda x: x.where(x != 30, 20)),
            # 放到當前裝置/型別
            EnsureTyped(keys=["image", "label"]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        data = {"image": self.image_files[idx], "label": self.label_files[idx]}
        return self.transforms(data)
