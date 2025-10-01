import torch
from monai.networks.nets import SwinUNETR

def get_swin_unetr_model(img_size=(128, 128, 128), in_channels=1, out_channels=26, feature_size=48):
    """
    建立一個 SwinUNETR 模型。

    Args:
        img_size (tuple): 輸入影像的大小，需與 Dataset 中的 target_shape 一致。
        in_channels (int): 輸入影像的通道數 (例如: T1, T2 雙模態影像就是 2)。
        out_channels (int): 輸出的通道數 (例如: 二元分割就是 1)。
        feature_size (int): Swin Transformer 主幹網路的特徵維度大小。

    Returns:
        torch.nn.Module: SwinUNETR 模型。
    """
    model = SwinUNETR(
        # img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=True, # 使用 Checkpoint 可以節省 GPU 記憶體
    )
    return model