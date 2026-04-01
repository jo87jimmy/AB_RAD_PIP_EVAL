"""
MVTec DRAEM 測試資料集載入器
=============================

本模組提供測試資料集的載入功能，用於消融實驗推論。
僅包含推論所需的資料集類別，已移除訓練相關的資料增強邏輯。

資料集結構：
  mvtec/
    └── {category}/
        ├── test/
        │   ├── good/          ← 正常圖像
        │   ├── broken_large/  ← 異常圖像 (依缺陷類型分子目錄)
        │   └── ...
        └── ground_truth/
            ├── broken_large/  ← 對應的異常遮罩
            └── ...
"""

import os
import glob

import numpy as np
from torch.utils.data import Dataset
import torch
import cv2


class MVTecDRAEMTestDataset(Dataset):
    """
    MVTec DRAEM 測試資料集。

    載入測試圖像及其對應的 Ground Truth 異常遮罩。
    支援正常 (good) 和多種缺陷類型的圖像。

    原因：
      此資料集用於計算推論指標 (AUROC, PR-AUC 等)，
      需要同時提供預測輸入和 Ground Truth 標籤。
    """

    def __init__(self, root_dir, resize_shape=None):
        """
        初始化測試資料集。

        Args:
            root_dir (str): 測試目錄路徑 (如 './mvtec/bottle/test')
            resize_shape (list[int]): 圖像縮放目標尺寸 [H, W]
        """
        self.root_dir = root_dir
        # 搜尋所有子目錄下的 png 圖像
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        """
        載入並前處理圖像與遮罩。

        處理流程：
          1. 以 BGR 模式讀取圖像
          2. 以灰階模式讀取遮罩（若無則生成全零遮罩）
          3. 調整尺寸至目標大小
          4. 歸一化至 [0, 1] 範圍
          5. 轉置為 (C, H, W) 格式

        Args:
            image_path (str): 圖像檔案路徑
            mask_path (str or None): 遮罩檔案路徑 (None 表示正常樣本)

        Returns:
            tuple: (image [3, H, W], mask [1, H, W]) 皆為 float32 numpy 陣列
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # 正常樣本沒有異常遮罩，建立全零遮罩
            mask = np.zeros((image.shape[0], image.shape[1]))

        if self.resize_shape is not None:
            image = cv2.resize(
                image, dsize=(self.resize_shape[1], self.resize_shape[0])
            )
            mask = cv2.resize(
                mask, dsize=(self.resize_shape[1], self.resize_shape[0])
            )

        # 歸一化到 [0, 1]
        image = image / 255.0
        mask = mask / 255.0

        image = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        mask = (
            np.array(mask)
            .reshape((mask.shape[0], mask.shape[1], 1))
            .astype(np.float32)
        )

        # (H, W, C) → (C, H, W)，符合 PyTorch 格式
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        """
        取得單一測試樣本。

        根據圖像所在子目錄判斷是正常或異常樣本：
          - 'good' 目錄 → 正常 (has_anomaly=0)
          - 其他目錄 → 異常 (has_anomaly=1)，載入對應的 ground_truth 遮罩

        Args:
            idx (int): 樣本索引

        Returns:
            dict: 包含 'image', 'has_anomaly', 'mask', 'idx' 的字典
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)

        if base_dir == "good":
            # 正常樣本：無異常遮罩
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            # 異常樣本：載入對應的 ground_truth 遮罩
            # 遮罩路徑結構：../../ground_truth/{defect_type}/{filename}_mask.png
            mask_path = os.path.join(dir_path, "../../ground_truth/")
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {
            "image": image,
            "has_anomaly": has_anomaly,
            "mask": mask,
            "idx": idx,
        }
        return sample
